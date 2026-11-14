# Underexpanded Jet: Method of Characteristics

Python implementation of the method of characteristics for a planar, underexpanded supersonic jet leaving a nozzle. The code computes the wave pattern, the jet boundary, and the Mach number and static pressure inside the jet, up to the point where a shock forms.

![Wave pattern](images/wave_pattern.png)

## What it does

When a jet leaves a nozzle at a pressure above ambient ($p_e > p_a$), it expands at the nozzle lip through a Prandtl-Meyer fan. This code follows that expansion through the first jet cell:

1. **Fan (AA):** $\Gamma^-$ characteristics from the lip, between the two bounding rays.
2. **First centreline reflection (BB):** the fan rays reflect off the axis as $\Gamma^+$ waves.
3. **Jet boundary reflection (CC):** the BB waves reflect off the constant-pressure boundary as $\Gamma^-$ compression waves.
4. **Second centreline reflection (DD):** the CC waves reflect off the axis again as $\Gamma^+$ waves.
5. **Shock onset:** the DD waves are compression waves, and where they cross, the isentropic solution stops being valid. The onset is estimated from the order in which the DD waves arrive at the (extended) jet boundary compared with the order in which they left the centreline.

Mach number and pressure are then plotted as coloured fields up to the shock onset.

## Inputs

Set at the top of the script:
pa: ambient pressure [Pa] | 101325 
pe: exit pressure [Pa] | 2*pa
M_e: exit Mach number [-] | 1.5 
gamma: specific heat ratio [-] | 1.4 
height_full: full nozzle height [m] (`h_lip` is half of it) | 1 
phi_e: exit flow angle [rad] | 0 
n_ray: number of fan rays, bounding rays included (≥ 2) | 10 

## Requirements

Python 3 with:
- NumPy
- SciPy
- Matplotlib

## Usage
This prints the estimated shock position and opens three figures:

1. Mach number field
2. Static pressure ratio field
3. Wave pattern with the jet boundary and shock onset

## Method in short

- The flow is assumed planar, inviscid, steady and homentropic, so the total pressure is constant up to the shock. The jet boundary is a constant-pressure streamline.
- The Riemann invariants $V^- = \phi + \nu$ (along $\Gamma^-$) and $V^+ = \phi - \nu$ (along $\Gamma^+$) give the flow state at every wave crossing. $\nu$ is the Prandtl-Meyer angle, and its inverse is solved numerically.
- Wave positions use straight segments with the average of the slopes at the two ends (first order).
- Crossings are stored in $n_{ray} \times n_{ray}$ arrays, where the row is the newly reflected wave, the column is the wave it crosses, and the diagonal is the reflection point.

## Results for the default case

![Mach number](images/mach_field.png)
![Pressure ratio](images/pressure_field.png)

## Limitations

- **Valid only up to the shock onset.** Beyond it the total pressure is no longer constant, and the code does not compute the shock itself, so there is no solution past that point.
- **Approximate shock position.** It is found from straight-line extensions of the last DD segments and an extended boundary, so crossings before the last CC wave are not detected, and the estimate improves with larger `n_ray`.
- **Coloured fields are interpolated** linearly between the computed points, so they approximate the true distribution, especially in the fan and near the lip.
- First-order scheme, so accuracy depends on `n_ray`.

## Repository structure

```
.
├── jet_moc.py          # solver and plotting
├── images/             # figures used in this README
└── README.md
```

## Author

Kevin Wibowo, TU Delft
