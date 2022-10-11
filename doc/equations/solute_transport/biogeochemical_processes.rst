Biogeochemical processes
========================

Soil temperature
----------------

.. math::
  TA_{soil}=TA_{year} + A \cdot e^(\frac{-0.5 \cdot z_{soil}}{D}) \cdot sin(\omega t + \Phi -\frac{0.5 \cdot z_{soil}}{D})


.. math::
  \omega = \frac{2\pi}{365}

Denitrification
---------------

.. math::
  Denit_{soil}=\begin{cases}
  (Denit_{soil-max} \cdot \frac{NO_{3-soil}(T,t)}{k_{denit-soil} + NO_{3-soil}(T,t)}) \cdot c_{denit} & S_{t} \geq 0.7 \cdot S_{sat}\\
  0 & S_{t} < 0.7 \cdot S_{sat}
  \end{cases}

  with

  c_{denit}=\begin{cases}
  \frac{TA_{soil}}{50 - 5} & 5 \geq TA_{soil} \leq 50 \\
  0 & 5 > TA_{soil} or TA_{soil} > 50
  \end{cases}

.. math::
  Denit_{gw}=Denit_{gw-max} \cdot \frac{NO_{3-gw}(T,t)}{k_{denit-gw} + NO_{3-gw}(T,t)}

Nitrification
-------------

.. math::
  Nit_{soil}=\begin{cases}
  Nit_{soil}=(Nit_{soil-max} \cdot \frac{N_{min}(T,t)}{k_{nit} + N_{min}(T,t)}) \cdot c_{nit} & S_{t} < 0.7 \cdot S_{sat}\\
  0 & S_{t} > 0.7 \cdot S_{sat}
  \end{cases}

with

c_{nit}=\begin{cases}
\frac{TA_{soil}}{50 - 5} & 5 \geq TA_{soil} \leq 50 \\
0 & 5 > TA_{soil} or TA_{soil} > 50
\end{cases}

Nitrogen mineralization
-----------------------

.. math::
  Nmin_{soil}=k_{min} \cdot c_{min}

with

c_{min}=\begin{cases}
\frac{TA_{soil}}{50 - 5} & 5 \geq TA_{soil} \leq 50 \\
0 & 5 > TA_{soil} or TA_{soil} > 50
\end{cases}


Nitrogen uptake by crops
------------------------
.. math::
  C_{up}=\int_{T=0}^{\infty} C_S(T,t) \cdot \alpha_p \cdot \overleftarrow{p}_{transp}(T,t) dT
