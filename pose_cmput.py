import numpy as np
import os
from numpy import polyfit,poly1d
def test(p_path,t_path,mode="line"):
    if os.path.exists(p_path):
        if mode == "line":
            pose_line = np.array(np.load(p_path)['arr_0'])
            t_past = np.array(np.load(t_path)['arr_0'])
            t_past = t_past - t_past[0]
            x_past = pose_line[:,0]
            y_past = pose_line[:,1]

            print(pose_line[:,0],t_past)
            print(pose_line.size,t_past.size)

            coeff_x =  polyfit(t_past, x_past,1)
            coeff_y =  polyfit(t_past, y_past,1)
            print(coeff_x,coeff_y)
            fx = poly1d(coeff_x)
            fy = poly1d(coeff_y)
            x = fx(t_past[-1]+20)
            y = fy(t_past[-1]+20)

            print(x,y)
def main():
    p_path = "pose_path00078.npz"
    t_path = "pose_path_t00078.npz"
    test(p_path,t_path)

if __name__ == "__main__":
    main()