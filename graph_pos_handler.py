import numpy as np
 
def save_pos(file_name: str, pos:dict) -> None:
    '''
    save position info to npy file
    '''
    np.save(file_name, pos)

def load_pos(file_name: str) -> dict:
    '''
    load position info from npy file and convert to dict
    '''
    pos_np = np.load(file_name, allow_pickle=True)
    pos = pos_np.reshape(1,)[0]
    return pos