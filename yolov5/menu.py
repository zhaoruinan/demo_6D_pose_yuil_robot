import wx
import time
import numpy as np
import socket
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 8089  # The port used by the server
def robot_read_pos():
     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        data = s.recv(1024)
        s.sendall(b"read pos")
        data = s.recv(1024)
        #print(f"Received {data!r}")
def robot_set_pos(pose = None,home = False,stop = False,axis = 1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        if home:
            s.sendall(b"HOME")
        elif stop:
            s.sendall(b"STOP")
        else:
            pose_b = pose.tobytes()
            if axis ==1:
                s.sendall(b"PS_T"+pose_b)
            else:
                s.sendall(b"PS_J"+pose_b)
        data = s.recv(1024)
        pose_j = np.frombuffer(data, dtype=np.float64)
        #print(f"Received {data!r}")
        #print(pose_j)

class MyApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(parent = None,title = 'wxPython',size = (500,400))
        panel = wx.Panel(frame, -1)

        self.pose = np.array([0.601, -0.173, 0.277,-3.14, 0.0, 1.56])
        self.pose_j = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.button_xm = wx.Button(panel,-1,'x-',pos = (30,40))
        self.button_ym = wx.Button(panel,-1,'y-',pos = (30,80))
        self.button_zm = wx.Button(panel,-1,'z-',pos = (30,120))
        self.Bind(wx.EVT_BUTTON, self.OnButton_xm,self.button_xm)
        self.Bind(wx.EVT_BUTTON, self.OnButton_ym,self.button_ym)
        self.Bind(wx.EVT_BUTTON, self.OnButton_zm,self.button_zm)

        self.button_xp = wx.Button(panel,-1,'x+',pos = (120,40))
        self.button_yp = wx.Button(panel,-1,'y+',pos = (120,80))
        self.button_zp = wx.Button(panel,-1,'z+',pos = (120,120))
        self.Bind(wx.EVT_BUTTON, self.OnButton_xp,self.button_xp)
        self.Bind(wx.EVT_BUTTON, self.OnButton_yp,self.button_yp)
        self.Bind(wx.EVT_BUTTON, self.OnButton_zp,self.button_zp)


        self.button_1m = wx.Button(panel,-1,'1-',pos = (210,40))
        self.button_2m = wx.Button(panel,-1,'2-',pos = (210,80))
        self.button_3m = wx.Button(panel,-1,'3-',pos = (210,120))
        self.button_4m = wx.Button(panel,-1,'4-',pos = (210,160))
        self.button_5m = wx.Button(panel,-1,'5-',pos = (210,200))
        self.button_6m = wx.Button(panel,-1,'6-',pos = (210,240))
        self.Bind(wx.EVT_BUTTON, self.OnButton_1m,self.button_1m)
        self.Bind(wx.EVT_BUTTON, self.OnButton_2m,self.button_2m)
        self.Bind(wx.EVT_BUTTON, self.OnButton_3m,self.button_3m)
        self.Bind(wx.EVT_BUTTON, self.OnButton_4m,self.button_4m)
        self.Bind(wx.EVT_BUTTON, self.OnButton_5m,self.button_5m)
        self.Bind(wx.EVT_BUTTON, self.OnButton_6m,self.button_6m)
        self.button_auto = wx.Button(panel,-1,'auto',pos = (210,280))

        self.button_1p = wx.Button(panel,-1,'1+',pos = (300,40))
        self.button_2p = wx.Button(panel,-1,'2+',pos = (300,80))
        self.button_3p = wx.Button(panel,-1,'3+',pos = (300,120))
        self.button_4p = wx.Button(panel,-1,'4+',pos = (300,160))
        self.button_5p = wx.Button(panel,-1,'5+',pos = (300,200))
        self.button_6p = wx.Button(panel,-1,'6+',pos = (300,240))
        self.button_home = wx.Button(panel,-1,'home',pos = (300,280))
        self.Bind(wx.EVT_BUTTON, self.OnButton_1p,self.button_1p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_2p,self.button_2p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_3p,self.button_3p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_4p,self.button_4p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_5p,self.button_5p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_6p,self.button_6p)
        self.Bind(wx.EVT_BUTTON, self.OnButton_home,self.button_home)

        self.Bind(wx.EVT_BUTTON, self.OnButton_home,self.button_auto)


        frame.Show()
        return True
    def OnButton_auto(self, event):
        robot_set_pos(home = True)
    def OnButton_home(self, event):
        robot_set_pos(home = True)
    def OnButton_xm(self, event):
        self.pose[0] =  self.pose[0]-0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_ym(self, event):
        self.pose[1] =  self.pose[1]-0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_zm(self, event):
        self.pose[2] =  self.pose[2]-0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_xp(self, event):
        self.pose[0] =  self.pose[0]+0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_yp(self, event):
        self.pose[1] =  self.pose[1]+0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_zp(self, event):
        self.pose[2] =  self.pose[2]+0.01
        print(self.pose)
        robot_set_pos(pose = self.pose)
    def OnButton_1m(self, event):
        self.pose_j[0] =  self.pose_j[0]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_2m(self, event):
        self.pose_j[1] =  self.pose_j[1]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_3m(self, event):
        self.pose_j[2] =  self.pose_j[2]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_4m(self, event):
        self.pose_j[3] =  self.pose_j[3]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_5m(self, event):
        self.pose_j[4] =  self.pose_j[4]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_6m(self, event):
        self.pose_j[6] =  self.pose_j[5]-0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_1p(self, event):
        self.pose_j[0] =  self.pose_j[0]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_2p(self, event):
        self.pose_j[1] =  self.pose_j[1]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_3p(self, event):
        self.pose_j[2] =  self.pose_j[2]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_4p(self, event):
        self.pose_j[3] =  self.pose_j[3]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_5p(self, event):
        self.pose_j[4] =  self.pose_j[4]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
    def OnButton_6p(self, event):
        self.pose_j[5] =  self.pose_j[5]+0.01
        print(self.pose_j)
        robot_set_pos(pose = self.pose_j,axis = 0)
def menu():

    app = MyApp()
    app.MainLoop()

if __name__ == "__main__":
    menu()