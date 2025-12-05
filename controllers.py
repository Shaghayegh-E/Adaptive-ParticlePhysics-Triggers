"""Controller algorithms (PD controllers) for trigger rate control."""

def PD_controller1(r_: float, pre_: float, cut_: float):
    """PD controller variant used for HT cut adjustments.

    Returns (new_cut, error)
    """
    Kp = 30
    Kd = 5
    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp * error + Kd * delta
    return newcut_, error


def PD_controller2(r_: float, pre_: float, cut_: float):
    """PD controller variant used for AD (AS) cut adjustments.

    Returns (new_cut, error)
    """
    Kp = 15
    Kd = 0
    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp * error + Kd * delta
    return newcut_, error
