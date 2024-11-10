from automated_var.src.assistant_var_functionalities import mistaken_identity_checker as mic
from automated_var.src.assistant_var_functionalities import offside_checker as oc


class AssistantReferee(mic.IdentityCheck, oc.OffsideCheck):
    def __init__(self):
        pass