from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from simple_pid import PID
from xopt.generator import Generator


class PIDGenerator(Generator):
    # inputs when creating generator
    target_value: float
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    sim_time: float = 1.0  # set to 0 if not a simulation

    # internal variables
    pid: PID = None

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_variables != 1:
            raise ValueError("this generator only supports one variable")

        if v.n_observables != 1:
            raise ValueError("this generator only supports one observable")
        return v

    def generate(self, n_candidates) -> list[dict]:
        if self.pid is None:
            self.pid = PID(self.Kp, self.Ki, self.Kd, self.target_value)
            self.pid.output_limits = self.vocs.variables[self.vocs.variable_names[0]]

        if n_candidates != 1:
            raise NotImplementedError()

        # get the last value
        last_value = float(self.data[self.vocs.observable_names].iloc[-1, 0])

        # run pid controller
        if self.sim_time > 0:
            control_value = self.pid(last_value, self.sim_time)
        else:
            control_value = self.pid(last_value)

        return [{self.vocs.variable_names[0]: control_value}]
