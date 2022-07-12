# BSc Thesis

## Abstract

A hybrid system is composed of multiple energy sources. Those with weather
dependency represent uncertainty in demand provision, which could compromise the
functioning of the supplied services. Therefore, the system operator needs to be
able to predict the energy balance and decide the best control strategy.

Predicting multiple time series is a difficult task to be done analytically,
hence artificial intelligence has been used as alternative. Historical data has
to be processed but the models have great generalization capacity, adapting to
the scenario.

The present work aimed to study the feasibility of predictive analysis in hybrid
systems using machine learning. Wind, solar and diesel sources were addressed
along with energy storage. The data used was taken from the [SONDA] network,
provided by [INPE], at the Petrolina meteorological station.

Neural Network was used as supervised learning. The algorithm applied was the
_Recurrent Neural Network_. Python language was used with the help of libraries
like _numpy_, _sklearn_ and _tensorflow_.

Neural networks performed well in short-term horizons but degraded with longer
timesteps. The predictive strategy had accuracy of 73% and was able to obtain
similar results to traditional ones. There is still space for improvement in
future work.

## Compiling
```bash
arara --working-directory tex 0_tcc.tex
```
```
  __ _ _ __ __ _ _ __ __ _
 / _` | '__/ _` | '__/ _` |
| (_| | | | (_| | | | (_| |
 \__,_|_|  \__,_|_|  \__,_|

Processing "0_tcc.tex" (size: 4.2 kB, last modified: 07/12/2022
17:26:14), please wait.

(PDFLaTeX) PDFLaTeX engine .............................. SUCCESS
(Nomencl) The Nomenclature software ..................... SUCCESS
(Biber) The Biber reference management software ......... SUCCESS
(MakeGlossariesLite) The MakeGlossariesLite software..... SUCCESS
(PDFLaTeX) PDFLaTeX engine .............................. SUCCESS
(PDFLaTeX) PDFLaTeX engine .............................. SUCCESS
```

[SONDA]: <http://sonda.ccst.inpe.br/>
[INPE]: <https://www.gov.br/inpe/pt-br>
