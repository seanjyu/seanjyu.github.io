---
layout: "single"
author_profile: true
title: Beam Gravity Sytem Optimization Webapp
permalink: /beam_optimization_app/
published: true
---

After graduating from my masters in engineering I made this small app that optimizes the number of interior beams and sizes the members of the system based on a set of gravity loads. Although the app currently only takes dead and live loads but this function can easily be expanded to include other gravity loads. The AISC W-sections were used to size all the members, this could also be expanded to consider different types of sections. Originally the app was hosted on heroku, however heroku has ended their free service in 2022 so the app is now hosted on streamlit. The link to the app can be found <a href="" target="_blank"  rel="noopener noreferrer">here</a> and the github repository with the code can be found <a target="_blank" rel="noopener noreferrer" href="https://github.com/sjy2129/Beam_Opt_app" >here</a>. The following post goes through the algorithm of used to find an optimal gravity beam design.
<br><br>
The structural calculations were coded into various functions since the optimization was an iterative process.
Below is a brief explanation of each function used for the structural calculations.
<br>

<h5> Loading </h5>
Two functions (beam_load, girder_load) were created to calulate the load, one for the beams and one for the girder. The functions made use of simple static equations and concepts i.e. moment being the integral of shear diagram. Note the beam connections are assumed to be pinned.

<h5> Design </h5>

The functions design, shear_design, add_self_weight are used to size the beam based on AISC structural requirements. Below is a flowchart on how these functions are utilized. First the depth requirements are fulfilled. Then the moment strengths for each failure mode (Plastic, Elastic and Inelastic) are calculated for all of the possible sizes. The lightest section is seleted then is checked for shear and then checked again for both moment and shear after adding self-weight. If this section fulfills the demand requirements then it is used as the solution. If not it will loop back to check the next lightest section and its capacities until a satisfactory section is chosen.

![design_flowchart](\assets\images\posts\2023_01_28_beam_gravity_system_optimization\design_flowchart.png)

<h5>Optimization</h5>
The next function (frame_optimizer) of the code optimizes the number of infill beams. Once again, since the code is quite long so a flowchart of this function is shown below. Essentially, this function employs a recursive algorithm. The input for this function are the area loads and dimension of the frame. The function takes the inputs and finds the shorter side thus finding the length of the beam. Then the function first designs the system for a single infill beam and then just looks for a local optimum by incrementally increasing the number of infill beams and designing the system until the weight is larger than the previous system.

![optimization_flowchart](\assets\images\posts\2023_01_28_beam_gravity_system_optimization\optimization_flowchart.png)

<h5>Visualization</h5>
The last function visualizes the beam system by plotting them as lines on a graph. The graph was made using plotly. Then the rest of the code just creates the UI through streamlit and runs the functions I mentioned above. I haven't included the code in this post since it is relatively straight forward.

<h5>Future Considerations</h5>

As mentioned above, other kinds of gravity loads, such as snow or roof, can easily be added to the model. Similarly, considering moment connections would also be trivial. Another future consideration could be designing for dynamics and considering the natural frequency of the framing system.
