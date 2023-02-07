---
layout: "single"
author_profile: true
title: Preliminary Design of Tuned Mass Dampers Webapp
permalink: /tmd_preliminary_design/
usemathjax: true
published: true
---

The following is a short write up of a small app that provides preliminary design paramaters for a tuned mass damper. The algorithm used in the app is based off the methods described in the book "Structural Motion Engineering" by Jerome Connor and Simon Laflamme<sup>1</sup> (this book is really great for dynamical control in structural engineering for those who are interested). It is important to note that the output design variables are optimized to counter resonance and therefore may not be effective agaisnt high impulse loads.

<details>
    <summary>Background information on tuned mass dampers for non-engineers</summary>
        <h6>Quick introduction on why we need tuned mass dampers</h6>
        <p> Buildings are designed to resist many types of loads. Depending on the location and height of the building an engineer may need consider dynamical loading. Examples of dynamic loads are earthquakes which can last from a few seconds to half a minute and winds which can be sustained for hours during a typhoon or heavy rainstorm. The special thing about these loads is that the force which is applied changes over a period of time. Some dynamic loads can be considered periodic therefore can have a frequency. The issue is when the frequency of the load matches the natural frequency of the building causing resonance. The natural frequency can be defined as the frequency at which an object freely oscillates without any forces applied and all objects have one. Resonance causes the applied amplitude to drastically increase which could cause serious damage to a building. One solution to resonance would be tuned mass dampers. A tuned mass damper is designed to oppose resonance by being 90 degrees out of phase from the natural frequency of the building thereby decreasing the effects of resonance.</p>
</details>

<h5>Algorithm</h5>

In this app, the optimal design can be defined as the particular design variables that lead to a system in which the peak dynamic amplification factor (amplification resulting from ground motion/acceleration) is below a specified peak. These design variables include the mass, frequency and damping ratio of the tuned mass damper. The book "Structural Motion Engineering" lays out methodologies to find these variables for two cases, one where the building is assumed to be undamped and another where the building has damping<sup>\*</sup>. The assumptions for these methods are that the building is a single degree of freedom system (SDOF) with an additional mass attached to the top. <br><br> The only inputs for the algorithm are the maximum dynamic amplification response factor and the building damping ratio. However, sometimes if the dynamic factor criteria is too low the algorithm will not be able to find a solution, therefore the user will be asked to change the inputs. <br> <br> While trying to write this post I had originally inteded to go through how the equations were derived but I realized the textbook does that really well and I would just be copying the textbook so in this post I have summarized the important formula and explain how I have applied them to make this app.

<sup>\*</sup> These methods are largely based off the book "Mechanical Vibrations" by Jacob Pieter Den Hartog <sup>2</sup> and a paper by Tsai.H-C, and Lin. G-C <sup>3</sup>.

<h5>Undamped Building with TMD</h5>
The displacements of an undamped building with a tuned massed damper can be given in polar form by:

$$
\begin{aligned}
    \bar{u} &=\frac{\hat{p}}{k}H_1 e^{i\delta_1}-\frac{\hat{a}_g m}{k}H_2e^{i\delta_2} \\
    \bar{u}_d &=\frac{\hat{p}}{k_d}H_3 e^{-i\delta_3} - \frac{\hat{a}_g m}{k_d}H_4 e^{-i\delta_4}
\end{aligned}
$$

In this system we have applied harmonic loads for the ground accelleration and force applied to the primary mass(this means that cosine and sine inputs would correspond to real and imaginary parts). \\(\hat{a_g}\\) represents the amplitude of ground accelleration and \\(\hat{p}\\) is the amplitude of force applied to the primary mass. To optimize the design we are interested in finding the design variables such that the coefficient \\(H_2\\) (this is the dynamic amplification factor). The equation of \\(H_2\\) can be given by:

$$
\begin{aligned}
    H_2 &=\frac{\sqrt{\left[(1+\bar{m}) f^2-\rho^2\right]^2+\left[2 \xi_d \rho f(1+\bar{m})\right]^2}}{\left|D_2\right|} \\
    \left|D_2\right| &=\sqrt{\left[\left(1-\rho^2\right)\left(f^2-\rho^2\right)-\bar{m} \rho^2 f^2\right]^2+\left[2 \xi_d \rho f\left(1-\rho^2[1+\bar{m}]\right)\right]^2}
\end{aligned}
$$

Where:

$$
\begin{aligned}
    \bar{m} = \frac{m_d}{m} \\
    f = \frac{\omega_d}{\omega} \\
    \rho = \frac{\Omega}{\omega} \\

\end{aligned}
$$

and \\(m_d\\) is the mass of the damper, \\(m\\) is the mass of the building, \\(\omega_d\\) is the frequency of the damper, \\(\omega\\) is the frequency of the building, \\(\Omega\\) is the frequency of the exciting force. Therefore, \\(\bar{m}\\), \\(f\\) are the mass and frequency ratios of the damper to the building. <br><br>

The variables that are to be chosen by the designer are \\(\bar{m}\\), \\(f\\), \\(\xi_d\\) which brings up the question how can we choose an optimal solution for all three of these varibles? <br><br>

First lets see what happens if we set the mass ratio and the frequency ratio we create a plot of the frequency ratio and \\(H_2\\). The plot below fixes \\(\bar{m}\\) to 0.01 and \\(f\\) to 1 with the different lines representing the different damping ratios.

{% include h2plot.html %}

There are two key things to observe, firstly that as \\(\xi_d\\) increases the plot forms a single peak (see lines 1 and 0.1) and when the damping ratio is smaller two peaks form ( see lines 0.05 and 0.03). The second thing to observe is that each of the dynamic responses pass through the same 2 points. Turns out an optimal design can be found where each of these peaks can be the same and below a specified point. The book derives equations for the design variables to achieve this optimal design:

$$
\begin{align}
    H_{2|opt} & = \frac{1+\bar{m}}{\sqrt{0.5\bar{m}}} \\
    f_{opt} & = \frac{\sqrt{1-0.5 \bar{m}}}{1 + \bar{m}} \\
    \xi_{d|opt} & = \sqrt{\frac{\bar{m}(3 - \sqrt{0.5\bar{m}})}{8(1+\bar{m})(1-0.5\bar{m})}} \\
\end{align}
$$

Therefore, for the undamped building case if given a dynamic amplification criteria we can use a root finding algorithm on equation 1 to find an optimal solution for \\(\bar{m}\\) and then substitute the solution into equations 2 and 3 to find \\(f\_{opt}\\) and \\(\xi_d\\).

<h5>Damped Building with TMD</h5>
The method to find the optimal design parameters for a damped building is a little more complicated. Due to the inclusion of the damping ratio for the building there is no analytical solution for the optimal frequency ratio or mass ratio. This is because the dynamic response does not pass through the same 2 points like in the undamped case. In a paper by Tsai, H.-C. and Lin, G.-C. they find the optimal ratios through an iterative numerical procedure. First for a given building damping ratio the values for the mass ratio, TMD damping ratio and frequency ratio are assumed. Then the peak dynamic response is found (equivalent of \\(H_2\\) in the undamped case). This process is repeated for different values of ratios and the optimal solution are the variables that result in the minimum peak value of the dynamic response. Curve fitting schemes were then used to estimate the ratios on untested parameters. The maximum error (percentage difference from optimal solution) from the curve fitted equations was 2% and as low as 0.04$ whereas if the undamped equations were used the error would be over 20%. It is important to note that if the mass ratio is over 10% then the errors for optimal solution tend to be larger. <br><br>

The app uses these curve fitted equations to find the optimal solution. However the paper only did experiments on specific building damping ratios (0.02, 0.05 and 0.1), therefore an additional interpolation method, which was not included in the paper, had to be used.

<h5>Future works</h5>
There many other ways to find optimal parameters, I have seen others use evolutionary algorithms (artificial intelligence) <sup>4, 5</sup> to estimate parameters for tuned mass dampers. It could be interesting to see if these methods provide better solutions at higher mass ratios. However, these kinds of algorithms are not guaranteed to find a global optimum so I wonder if it would be possible to employ some sort of non-linear/linear optimization techniques. Maybe if additional design constraints could be added such as a maximum mass ratio. Another direction that could be worth looking into is the optimization criteria. The aforementioned methods were mainly optimized for the peak dynamic response under harmonic excitation. It could be worth looking into optimized parameters against high impact loads (although it can be argued if high impulse loads are an issue there maybe more effective measures).

<h5>Refrences</h5>
1) Connor, J. and Laflamme, S. (2014) Structural Motion Engineering. Springer International Publishing. <br>
2) Pieter, D.H.J. (1934) Mechanical vibrations. McGraw-Hill. <br>
3) Tsai, H.-C. and Lin, G.-C. (1993) “Optimum tuned-mass dampers for minimizing steady-state response of support-excited and damped systems,” Earthquake Engineering &amp; Structural Dynamics, pp. 957–973.<br>
4) Mohebbi, M. et al. (2012) “Designing optimal multiple tuned mass dampers using genetic algorithms (gas) for mitigating the seismic response of structures,” Journal of Vibration and Control, 19(4), pp. 605–625.<br>
5) Özsarıyıldız, Ş.S. and Bozer, A. (2014) “Finding optimal parameters of tuned mass dampers,” The Structural Design of Tall and Special Buildings, 24(6), pp. 461–475.
