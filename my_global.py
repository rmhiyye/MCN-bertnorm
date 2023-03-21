####################
# global value
####################

def _init(): # initialize
    global _global_dict
    _global_dict = {}
 
 
def set_value(key,value):
    """ define a global value """
    _global_dict[key] = value
 
 
def get_value(key,defValue=None):
    """ return a global value """
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
