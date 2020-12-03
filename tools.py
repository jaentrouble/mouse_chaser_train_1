import numpy as np
GRAVITY_C = 50

def gravity(rr, cc, last_rc = None):
    if last_rc is None:
        last_rc = np.array([rr[0],cc[0]])
    rcrc = np.stack([rr,cc],axis=1)
    new_rcrc = []
    for rc in rcrc:
        delta = rc - last_rc
        dist = np.linalg.norm(delta)
        if dist > GRAVITY_C:
            new_delta = delta * (GRAVITY_C**2)/(dist**2)
            new_rc = (last_rc + new_delta).astype(np.int)
            new_rcrc.append(new_rc)
            last_rc = new_rc
        else:
            new_rcrc.append(rc)
            last_rc = rc
    new_rr, new_cc = np.array(new_rcrc).swapaxes(0,1)
    return new_rr, new_cc, last_rc
