---
layout: "single"
author_profile: true
title: Preliminary Design of Tuned Mass Dampers Webapp
permalink: /tmd_preliminary_design/
usemathjax: true
published: false
---

The following is a short write up of a small app that provides preliminary design paramaters for a tuned mass damper. The algorithm used in the app is based off the methods described in the book "Structural Motion Engineering" by Jerome Connor and Simon Laflamme<sup>1</sup> (this book is really great for dynamical control in structural engineering for those who are interested). It is important to note that the output design variables are optimized to counter resonance and therefore may not be effective agaisnt high impulse loads.

<details>
    <summary>Background information on tuned mass dampers for non-engineers</summary>
        <h6>Quick introduction on why we need tuned mass dampers</h6>
        <p> Buildings are designed to resist many types of loads. Depending on the location and height of the building an engineer may need consider dynamical loading. Examples of dynamic loads are earthquakes which can last from a few seconds to half a minute and winds which can be sustained for hours during a typhoon or heavy rainstorm. The special thing about these loads is that the force which is applied changes over a period of time. Some dynamic loads can be considered periodic therefore can have a frequency. The issue is when the frequency of the load matches the natural frequency of the building causing resonance. The natural frequency can be defined as the frequency at which an object freely oscillates without any forces applied and all objects have one. Resonance causes the applied amplitude to drastically increase which could cause serious damage to a building. One solution to resonance would be tuned mass dampers. A tuned mass damper is designed to oppose resonance by being 90 degrees out of phase from the natural frequency of the building thereby decreasing the effects of resonance.</p>
</details>

<h5>Algorithm</h5>

In this app, the optimal design can be defined as the particular design variables that lead to a system in which the peak dynamic amplification factor (amplification resulting from ground motion/acceleration) is below a specified peak. These design variables include the mass, frequency and damping ratio of the tuned mass damper. The book "Structural Motion Engineering" lays out methodologies to find these variables for two cases, one where the building is assumed to be undamped and another where the building has damping<sup>\*</sup>. The assumptions for these methods are that the building is a single degree of freedom system (SDOF) with an additional mass attached to the top. <br><br> The only inputs for the algorithm are the maximum dynamic amplification response factor and the building damping ratio. However, sometimes if the dynamic factor criteria is too low the algorithm will not be able to find a solution, therefore the user will be asked to change the inputs. <br>

<sup>\*</sup> These methods are largely based off the book "Mechanical Vibrations" by Jacob Pieter Den Hartog <sup>2</sup> and a paper by Tsai.H-C, and Lin. G-C <sup>3</sup>.

<h5>Undamped Building with TMD</h5>
The equations of motion can be given by for an <br>

$$
\begin{aligned}
    & m_d\ddot{u}_d+c_d\dot{u}_d+k_d u_d + m_d + \ddot{u} =-m_d a_g \\
    & m \ddot{u}+k u - c_d\dot{u}_d -k_d u_d=-m a_g+p
\end{aligned}
$$

Where \\(a_g\\) is the ground accelleration and \\(p\\) is a force applied to the primary mass. If we assume them to be complex quantities (this means that cosine and sine inputs would correspond to real and imaginary parts of \\(a_g\\) and \\(p\\) ) i.e. <br>

$$
\begin{aligned}
    & a_g = \hat{a}_g e^{i\Omega t}\\
    & p = \hat{p} e^{i\Omega t}
\end{aligned}
$$

Note \\(\hat{a_g}\\) and \\(\hat{p}\\) are real quantities and correspond to the amplitude of the inputs.
The response can be described as: <br>

$$
\begin{aligned}
    & u = \bar{u}  e^{i\Omega t}\\
    & u_d = \bar{u}_d  e^{i\Omega t}
\end{aligned}
$$

where \\(\bar{u}\\) and \\(\bar{u}\_d\\) are complex quantities.
If we solve for the solution and convert it to polar form we are given the following:

$$
\begin{aligned}
    \bar{u} &=\frac{\hat{p}}{k}H_1 e^{i\delta_1}-\frac{\hat{a}_g m}{k}H_2e^{i\delta_2} \\
     \hat{u}_d &=\frac{\hat{p}}{k_d}H_3 e^{-i\delta_3} - \frac{\hat{a}_g m}{k_d}H_4 e^{-i\delta_4} \\
     H_1 &=\frac{\sqrt{\left(f^2-\rho^2\right)^2+\left(2 \xi_d \rho f\right)^2}}{\left|D_2\right|} \\
     H_2 &=\frac{\sqrt{\left[(1+\bar{m}) f^2-\rho^2\right]^2+\left[2 \xi_d \rho f(1+\bar{m})\right]^2}}{\left|D_2\right|} \\
     H_3 &=\frac{\rho^2}{\left|D_2\right|} \\
     H_4 &=\frac{1}{\left|D_2\right|} \\
     \left|D_2\right| &=\sqrt{\left[\left(1-\rho^2\right)\left(f^2-\rho^2\right)-\bar{m} \rho^2 f^2\right]^2+\left[2 \xi_d \rho f\left(1-\rho^2[1+\bar{m}]\right)\right]^2} \\
     \delta_1 &=\alpha_1-\delta_3 \\
     \delta_2 &=\alpha_2-\delta_3 \\
    \tan \delta_3 &=\frac{2 \xi_d \rho f\left[1-\rho^2(1+\bar{m})\right]}{\left(1-\rho^2\right)\left(f^2-\rho^2\right)-\bar{m} \rho^2 f^2} \\
     \tan \alpha_1 &=\frac{2 \xi_d \rho f}{f^2-\rho^2} \\
     \tan \alpha_2 &=\frac{2 \xi_d \rho f(1+\bar{m})}{(1+\bar{m}) f^2-\rho^2}
\end{aligned}
$$

The solution for the system is quite complicated, however to find an optimal solution we are mostly interested in

<h5>Damped Building with TMD</h5>
In a similar way to the method for the undamped building, we can

<h5>Future works</h5>
These

<h5>Refrences</h5>
1) Connor, J. and Laflamme, S. (2014) Structural Motion Engineering. Springer International Publishing. <br>
2) Pieter, D.H.J. (1934) Mechanical vibrations. McGraw-Hill. <br>
3) Tsai, H.-C. and Lin, G.-C. (1993) “Optimum tuned-mass dampers for minimizing steady-state response of support-excited and damped systems,” Earthquake Engineering &amp; Structural Dynamics, pp. 957–973.
