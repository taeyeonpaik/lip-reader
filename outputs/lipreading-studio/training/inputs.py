"""Normalize RGB LRS2 videos and pre-cropped grayscale LRS3 arrays identically."""
import cv2
import numpy as np
import torch


def load_pixels(row,device):
    if row.get('format')=='gray-npz':
        with np.load(row['path'],allow_pickle=False) as data:frames=data['frames']
        if frames.dtype!=np.uint8 or frames.shape!=(row['declared_frames'],96,96):
            raise ValueError('Invalid grayscale mouth crops')
        pixels=torch.from_numpy(frames[:,4:92,4:92].copy()).float()/255
    else:
        cap=cv2.VideoCapture(row['path']);frames=[]
        try:
            if abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01:raise ValueError('Wrong frame rate')
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame.shape!=(96,96,3):raise ValueError('Expected 96x96 RGB mouth crops')
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:cap.release()
        if len(frames)!=row['declared_frames'] or not frames:raise ValueError('Wrong frame count')
        rgb=torch.from_numpy(np.stack(frames)[:,4:92,4:92].copy()).float()/255
        pixels=rgb[...,0]*.2989+rgb[...,1]*.587+rgb[...,2]*.114
    if len(pixels)==0:raise ValueError('Empty mouth-crop sequence')
    return ((pixels-.421)/.165).unsqueeze(0).unsqueeze(2).to(device)
