"""functional_flow_climate: a neural-operator dynamics field in the physical function space of ERA5.

Pipeline (project goal): flow matching (pg-flow, W2 / Sobolev structure) -> ensemble scenarios ->
neural operator on the physical dynamics function space (this package) -> prediction, compared with a
climate-ODE baseline and scored with WeatherBench2.

Step 1 (this package): ``ffc.field.DynamicsField`` learns the tangent vector field dx/dt of the
66-channel KAI state as a spherical neural operator, trained on rollouts (L2 + Sobolev H^s loss) and on
the strong-form primitive-equation residual with its own tendency (PINN term). ``ffc.ensemble`` reads
and writes pg-flow ensemble files, ``ffc.evaluate`` scores them with WeatherBench2.
"""

from . import deps  # noqa: F401  (puts pgcfm and neuralop on sys.path)

__version__ = "0.1.0"
