import numpy as np


def log_wind_profile(wind_speed_2, height_1, height_2, surf_roughness=1):
    return wind_speed_2 * (
        (np.log(height_1) - np.log(surf_roughness))
        /
        (np.log(height_2) - np.log(surf_roughness))
    )


def wind_power(speed, curve=power_curve):
    try:
        return curve.loc[round(speed*2)/2]
    except KeyError:
        return curve.loc[round(speed*2+1)/2]


def solar_power(irradiation,
                modules,
                ambient_temp,
                NOCT_temp=45,
                ref_efficiency=0.1711,
                temp_coefficient=-0.003,
                ref_temp=25):

    pv_area = modules * 1.63
    cell_temp = ambient_temp + ((NOCT_temp - 20) * (irradiation / 800))
    efficiency = ref_efficiency * (1 - temp_coefficient * (cell_temp - ref_temp))

    return irradiation * pv_area * efficiency


def net_load(solar, wind, load):
    return solar + wind - load


def lolp(net_load):
    return (net_load < 0).sum()/len(net_load)


def load_following(net_load,
                   soc_previous,
                   diesel_timeout=False,
                   diesel_running=False,
                   soc_max=SOC_MAX,
                   dod_max=DOD_MAX,
                   diesel_power=DIESEL_POWER,
                   diesel_min=DIESEL_MIN):

    soc_min = soc_max * (1 - dod_max)
    diesel_fuel = 0
    dump = 0
    fuel = 0
    soc = soc_previous

    if diesel_timeout:
        diesel_power = 0

    if diesel_running:
        net_load += diesel_power
        fuel = diesel_power
        diesel_power = 0

    if soc_previous < soc_min or soc_previous > soc_max:
        raise ValueError(f'{soc_min} <! {soc_previous} <! {soc_max}')

    if net_load > 0:
        diesel_fuel = 0
        if net_load > soc_max - soc:
            dump = net_load - (soc_max - soc)
            soc = soc_max
            net_load = 0
        else:
            soc += net_load
            net_load = 0
    else:
        if -net_load > soc - soc_min:
            net_load += (soc - soc_min)
            soc = soc_min

            if -net_load > diesel_min:
                diesel_fuel = min(-net_load, diesel_power)
                net_load += diesel_fuel
            else:
                soc = min(soc_min+diesel_min+net_load, soc_max)
                diesel_fuel = diesel_min
                net_load = 0
        else:
            soc += net_load
            diesel_power = 0
            net_load = 0

    if diesel_running:
        diesel_fuel = fuel

    return net_load, soc, dump, diesel_fuel


def cycle_charging(net_load,
                   soc_previous,
                   diesel_timeout=False,
                   diesel_running=False,
                   soc_max=SOC_MAX,
                   dod_max=DOD_MAX,
                   diesel_power=DIESEL_POWER):

    soc_min = soc_max * (1 - dod_max)
    dump = 0
    fuel = 0
    diesel_fuel = 0
    soc = soc_previous

    if diesel_timeout:
        diesel_power = 0

    if diesel_running:
        net_load += diesel_power
        fuel = diesel_power
        diesel_power = 0

    if soc_previous < soc_min or soc_previous > soc_max:
        raise ValueError(f'{soc_min} <! {soc_previous} <! {soc_max}')

    if net_load > 0:
        diesel_fuel = 0
        if net_load > soc_max - soc:
            dump = net_load - (soc_max - soc)
            soc = soc_max
            net_load = 0
        else:
            soc += net_load
            net_load = 0
    else:
        if -net_load > soc - soc_min:
            net_load += (soc - soc_min)
            soc = soc_min

            if diesel_power > -net_load:
                diesel_power += net_load
                diesel_fuel = -net_load
                soc = min(diesel_power+soc, soc_max)
                diesel_fuel += soc - soc_min
                net_load = 0
            else:
                net_load += diesel_power
                diesel_fuel = diesel_power

        else:
            soc += net_load
            diesel_power = 0
            net_load = 0

    if diesel_running:
        diesel_fuel = fuel

    return net_load, soc, dump, diesel_fuel


def apply_dispatch(df,
                   cc_soc_start,
                   lf_soc_start,
                   diesel_lag,
                   soc_max=SOC_MAX,
                   dod_max=DOD_MAX,
                   diesel_power=DIESEL_POWER):

    d = df.copy(deep=True)

    d['lf_net_load'] = 0
    d['cc_net_load'] = 0
    d['lf_dump'] = 0
    d['cc_dump'] = 0
    d['lf_diesel'] = 0
    d['cc_diesel'] = 0
    d['lf_diesel_start'] = False
    d['cc_diesel_start'] = False
    d['lf_soc'] = lf_soc_start
    d['cc_soc'] = cc_soc_start
    d['lf_diesel_stop'] = True
    d['cc_diesel_stop'] = True
    lf_diesel_running = False
    cc_diesel_running = False
    lf_diesel_timeout = False
    cc_diesel_timeout = False

    for i in range(1, len(d)):
        lf_soc_previous = d['lf_soc'].iloc[i-1]
        cc_soc_previous = d['cc_soc'].iloc[i-1]
        lf_diesel_previous = d['lf_diesel'].iloc[i-1]
        cc_diesel_previous = d['cc_diesel'].iloc[i-1]
        net = d['net_load'].iloc[i]

        if i > diesel_lag and diesel_lag > 0:
            lf_diesel_running = (d['lf_diesel_start'].iloc[i-diesel_lag:i].sum() > 0)
            cc_diesel_running = (d['cc_diesel_start'].iloc[i-diesel_lag:i].sum() > 0)
            lf_diesel_timeout = (d['lf_diesel_stop'].iloc[i-diesel_lag:i].sum() > 0)
            cc_diesel_timeout = (d['cc_diesel_stop'].iloc[i-diesel_lag:i].sum() > 0)

        (net_lf,
         soc_lf,
         dump_lf,
         diesel_power_lf) = load_following(
             net,
             lf_soc_previous,
             soc_max=soc_max,
             dod_max=dod_max,
             diesel_timeout=lf_diesel_timeout,
             diesel_running=lf_diesel_running,
             diesel_power=lf_diesel_previous if lf_diesel_running else diesel_power
         )

        (net_cc,
         soc_cc,
         dump_cc,
         diesel_power_cc) = cycle_charging(
             net,
             cc_soc_previous,
             soc_max=soc_max,
             dod_max=dod_max,
             diesel_timeout=cc_diesel_timeout,
             diesel_running=cc_diesel_running,
             diesel_power=cc_diesel_previous if cc_diesel_running else diesel_power
        )



        d['lf_diesel_stop'].iloc[i] = (diesel_power_lf == 0 and lf_diesel_running)
        d['cc_diesel_stop'].iloc[i] = (diesel_power_cc == 0 and cc_diesel_running)
        d['lf_net_load'].iloc[i] = net_lf
        d['cc_net_load'].iloc[i] = net_cc
        d['lf_soc'].iloc[i] = soc_lf
        d['cc_soc'].iloc[i] = soc_cc
        d['lf_dump'].iloc[i] = dump_lf
        d['cc_dump'].iloc[i] = dump_cc
        d['lf_diesel_start'].iloc[i] = (diesel_power_lf > 0 and not lf_diesel_running)
        d['cc_diesel_start'].iloc[i] = (diesel_power_cc > 0 and not cc_diesel_running)
        d['lf_diesel'].iloc[i] = diesel_power_lf
        d['cc_diesel'].iloc[i] = diesel_power_cc

    return d
