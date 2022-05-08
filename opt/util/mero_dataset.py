# Extended NSVF-format dataset loader
# This is a more sane format vs the NeRF formats

from .util import Rays, Intrin, similarity_from_cameras
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import os
import cv2
import imageio
from tqdm import tqdm
import json
import numpy as np
from warnings import warn


class MeRoDataset(DatasetBase):
    """
    Extended NSVF dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __loadBbFromMgdata(self):
        #Meshing_
        for key in self.mgdata['graph'].keys():
            if key.startswith('Meshing_'):
                dmeshing = self.mgdata['graph'][key]
                if dmeshing['inputs']['useBoundingBox'] == False:
                    print("F*Ck0FF. dataset contains no Bounding box. I was so not made for this!\n come back when you added one for the mesh in meshroom!")
                    exit()
                bb = dmeshing['inputs']['boundingBox']
                x,y,z = bb['bboxTranslation'].values()
                #we are boldly ignoring rotation for now
                h,p,r = bb['bboxRotation'].values()
                #we are also ignoring individual bb-dimensions and use the radius instead. or the max dimensions, we'll see which one is more useful.
                sx,sy,sz = bb['bboxScale'].values()
                from math import sqrt
                radius = sqrt(sx**2+sy**2+sz**2)/2.
                maxdim = max(sx,sy,sz)
                self.scene_bsphere = [x,y,z,maxdim] #maybe maxdim .5 ?
                self.scene_bbox = torch.from_numpy(np.array([[x-maxdim,y-maxdim,z-maxdim],[x+maxdim,y+maxdim,z+maxdim]],dtype=np.float32))
                self.scene_scale = 1./maxdim
                self.scene_origin = [x,y,z]
    
    def __loadMainDataFile(self):
        #self.root_dir
        import json 
        #self.root_dir
        for ifile in os.listdir(self.root_dir):
            print(ifile)
            if ifile.endswith(".mg"):
                dfile = os.path.join(self.root_dir, ifile)
                print(dfile)
                with open(dfile,"r") as f:
                    data = f.read()
                    self.mgdata = json.loads(data)
                    return
        print("mgdata not found!")
    
    
    def __read_meta(self):
        
        ###
        ##    ##need poses, images, and camera intrinsics
        #we need img_files = [files to load].. ah a lot of stuff...
        
        ##maybe we should just randomly train and evaluate images on the fly instead of fixed sets.maybe not. who knows.
        sfm1 = self.mgdata['graph']['StructureFromMotion_1'] #we should check if it is _1. but hey.
        sfmuid  = sfm1['uids']['0'] #let's just hope it's there.
        sfmnodetype = sfm1['nodeType'] #nodetype-string, we need this for the correct path.
        #inpts = cint1['inputs']
        cdictname = "MeshroomCache"
        
        import json  #again yeah.
        #self.root_dir
        sfmfpath = os.path.join(self.root_dir,cdictname,sfmnodetype,sfmuid,"cameras.sfm") #bit more hardcoding. which is ok unless you fkd up your meshroom stuff for good.
        self.camerasfmdata = None
        with open(sfmfpath,"r") as f:
            data = f.read()
            self.camerasfmdata = json.loads(data)
            print("loaded sfm camera data")
                    #return
        #print("mgdata not found!")
        if not self.camerasfmdata:
            print("wtf, camera data not loaded?")
            exit()
        
        camintrinsics = self.camerasfmdata['intrinsics']
        self.camintrinsics = {}
        for intr in camintrinsics:
            intrinsicId= intr['intrinsicId']
            print(intrinsicId)
            self.camintrinsics[intrinsicId] = intr
        
        
        camposes = self.camerasfmdata['poses']
        self.camposes = {}
        for pose in camposes:
            poseID= pose['poseId']
            print(poseID)
            self.camposes[poseID] = pose['pose']
        
        print(self.camposes)
        
        #with open(os.path.join(self.root_dir, "intrinsics.txt")) as f:
        #    focal = float(f.readline().split()[0])
        ##did you actually hardcode ... everything? bro...i wondered why the heck none of my datasets worked.
        #focal = 300
        #self.intrinsics = np.array([[focal,0,400.0],[0,focal,400.0],[0,0,1]])
        #print(self.intrinsics)
        #self.intrinsics[:2] *= (np.array(self.img_wh)/np.array([800,800])).reshape(2,1)
        #print(self.intrinsics )
        ##srsly...
        #return
        #we don't even need the intrinsics outside this class. probably not even outside this method.
        
        #let's skipp that crap. jump straight to views
        
        #pose_files = sorted(os.listdir(os.path.join(self.root_dir, 'pose')))
        #img_files  = sorted(os.listdir(os.path.join(self.root_dir, 'rgb')))

        #if self.split == 'train':
        #    pose_files = [x for x in pose_files if x.startswith('0_')]
        #    img_files = [x for x in img_files if x.startswith('0_')]
        #elif self.split == 'val':
        #    pose_files = [x for x in pose_files if x.startswith('1_')]
        #    img_files = [x for x in img_files if x.startswith('1_')]
        #elif self.split == 'test':
        #    test_pose_files = [x for x in pose_files if x.startswith('2_')]
        #    test_img_files = [x for x in img_files if x.startswith('2_')]
        #    if len(test_pose_files) == 0:
        #        test_pose_files = [x for x in pose_files if x.startswith('1_')]
        #        test_img_files = [x for x in img_files if x.startswith('1_')]
        #    pose_files = test_pose_files
        #    img_files = test_img_files

        # ray directions for all pixels, same for all images (same H, W, focal)
        #time to go for views.
        
        
        #self.render_path = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        #print(self.render_path)
        
        """
          for img_fname in tqdm(img_files):
            img_path = path.join(root, img_dir_name, img_fname)
            image = imageio.imread(img_path)
            pose_fname = path.splitext(img_fname)[0] + ".txt"
            pose_path = path.join(root, pose_dir_name, pose_fname)
            #  intrin_path = path.join(root, intrin_dir_name, pose_fname)

            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            if len(cam_mtx) == 3:
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            full_size = list(image.shape[:2])
            rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
            if dynamic_resize:
                image = cv2.resize(image, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_gt.append(torch.from_numpy(image))


        self.c2w_f64 = torch.stack(all_c2w)
        
        """
        
        views = self.camerasfmdata['views']
        
        #dummyRootNP = NodePath("dummyRoot")
        
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.camdists = []
        temprays = []
        numviews=0
        totalidx = 0
        for view in views:
            numviews +=1
            if 'train' == self.split:
                if numviews > len(views)-3: #we can use the last 3 for training too. but then it's not independent.
                    pass
                    continue
            else:
                if numviews <= len(views)-3:
                    continue
            pose = self.camposes[view['poseId']]
            intr = self.camintrinsics[view['intrinsicId']]
            imgpath = view['path']
            w,h = int(view['width']), int(view['height'])
            f = float(intr["pxFocalLength"])
            cx, cy = intr['principalPoint']
            cx, cy = float(cx),float(cy)
            
            self.all_intr.append( Intrin(f, f, cx, cy) )
            
            distortionparams = intr['distortionParams'] #  meshroom spits out k1,k2,k3  but opencv and our undistort 5 wants k1 k2 p1 p2 k3
            distlist = []
            for dparam in distortionparams:
                distlist.append(float(dparam))
            #we use undistort with 0 params. just fill up to 5 all the time. 
            while len(distlist) <5:
                distlist.append(0)
            self.camdists.append([distlist[0],distlist[1],distlist[2]])
            #distlist2 = []
            #distlist2.append(distlist[0],distlist[1],distlist[3],distlist[4],distlist[2])
            #distlist = distlist2
            rot = pose['transform']['rotation']
            pos = pose['transform']['center']
            
            #dummyNP = dummyRootNP.attachNewNode("dummynp")
            xoff, yoff, zoff = self.scene_origin
            self.scene_scale
            xi,yi,zi = float(pos[0]),float(pos[1]),float(pos[2])
            
            xi = (xi-xoff)* self.scene_scale
            yi = (yi-yoff)* self.scene_scale
            zi = (zi-zoff)* self.scene_scale
            #self.center 
            #self.near_far = [0.1,5.0]#TODO: calculate near and far based on cameras , boundingbox and radius
            
            #panda needs the matrices transposed so yeah, indexing might appear a bit weirdo and is probably different if you use other libraries.
            #in that case you _may_ have more luck with the line provided below.
            #non-panda3d matvals = [ rot[0],rot[1],rot[2],pos[0], rot[3],rot[4],rot[5],pos[1], rot[6],rot[7],rot[8],pos[2], 0,0,0,1]
            matvals = [ rot[0],rot[3],rot[6],0, rot[1],rot[4],rot[7],0, rot[2],rot[5],rot[8],0, xi,yi,zi,1] #apply scene scale and offset here, sucks but that's svox2 base requirement
            matvalsf = []
            for xmat in matvals:
                matvalsf.append(float(xmat))
            
            #tensor([[ 8.8112e-01,  4.7053e-01, -4.7230e-02,  7.4131e-02],
            #[ 4.7290e-01, -8.7671e-01,  8.8002e-02, -1.3842e-01],
            #[ 3.2000e-08, -9.9875e-02, -9.9500e-01,  2.0632e+00],
            #[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
            #print(matvalsf)
            c2w = torch.FloatTensor([matvalsf[0:4],matvalsf[4:8],matvalsf[8:12],matvalsf[12:16]])
            c2w = torch.transpose(c2w,0,1)
            #print (c2w)
            #exit()
            self.poses.append(c2w)  # C2W
            
            #dummyNP.setMat(LMatrix4f(*matvalsf))
            #let's stick with the original image loading code. should be reasonably fine.
            

            image = imageio.imread(imgpath)
            self.all_gt.append(torch.from_numpy(image))
            self.all_c2w.append(c2w)
            #print("c2w",self.all_c2w[-1])
          
            
            ##this section needs work...
            #c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname)) #@ self.blender2opencv
            #c2w = torch.FloatTensor(c2w)
            ##self.poses.append(c2w)  # C2W #let's try to get away without
            
            #  get_rays outputs:
            #rays_o: (H*W, 3), the origin of the rays in world coordinate
            #rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
            #ray_o1 = dummyNP.getPos() #waste of memory to copy the ray origin gazillion of times. but yeah. can't fix rome in a night.
            #code below ain't a prime example of efficiency, it's more a proof of concept too.
            #ray_o1 = [ray_o1[0],ray_o1[1],ray_o1[2]]
            #rays_o = []
            
            #bodydata=GeomVertexData("body vertices", GeomVertexFormat.getV3c4() , Geom.UHStatic)
            #bodydata.setNumRows(h*w)
            #vertex = GeomVertexWriter(bodydata, 'vertex')
            #color = GeomVertexWriter(bodydata, 'color')
            #lastpercent = 0
            #pixidx = 0
            
            ###faster way
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32) + 0.5, #not sure about the +0.5 technically should get you the center of the pixels i guess? but for real cameras?
                torch.arange(w, dtype=torch.float32) + 0.5,
            )
            xx = (xx - cx) / f #could be fx and fy, if we had both separate.
            yy = (yy - cy) / f
            zz = torch.ones_like(xx)
            
            
            #using radial3 model.
            r2 = xx*xx + yy*yy
            r4 = r2*r2
            r6 = r4*r2
            r =  torch.ones_like(xx) + distlist[0]*r2 + distlist[1]*r4 + distlist[2]*r6
                
            #let's skip p1 and p2 parameters
            #dx = 2*p1*x0*y0 + p2*(r2 + 2*x0*x0) 
            #dy = p1*(r2 + 2*y0*y0) + 2*p2*x0*y0
            xx = xx*r#+dx
            yy = yy*r#+dy
            #return np.array((x * fx + cx, y * fy + cy))
            #print(xx,yy)
            
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(1, -1, 3, 1)
            dirs = (c2w[None,:3,:3]@dirs)[...,0][0]
            del xx, yy, zz
            #print(dirs)
            
            origins = torch.tensor([xi,yi,zi])
            origins = origins.repeat(h*w,1)
            trays = torch.cat((origins,dirs),1)
            #print(trays)
            self.all_rays.append(trays)
            #exit()
            #print((c2w[None,:3,:3]@dirs)[...,0][0])
            #exit()
            #dirs = (c2w[:, None, :3, :3] @ dirs)[..., 0]
            
                
            """
            for py in range(h):
                if int(10*py/h) != lastpercent:
                    lastpercent = int(10*py/h)
                    print(lastpercent,numviews)
                for px in range(w):
                    #print(px)
                    #let's hope we do row-collum-origin the right way.
                    #meshroom spits out k1,k2,k3  but opencv and our undistort 5 wants k1 k2 p1 p2 k3
                    distx, disty = self.distort5(px, py, cx, cy, f,f, distlist[0], distlist[1], distlist[3], distlist[4],distlist[2])
                    #print(distx,px,disty,py)
                    #worldCoordsViewVec = dummyRootNP.getRelativeVector(dummyNP,LVecBase3(px-cx, py-cy, f)).normalized()  #because f is the same for x and y we don't have to divide x and y but we can use z instead
                    worldCoordsViewVec = dummyRootNP.getRelativeVector(dummyNP,LVecBase3(distx-cx, disty-cy, f)).normalized()  #because f is the same for x and y we don't have to divide x and y but we can use z instead
                    temprays.append([ray_o1[0],ray_o1[1],ray_o1[2],worldCoordsViewVec[0],worldCoordsViewVec[1],worldCoordsViewVec[2]])
                    
                    vertex.addData3(ray_o1[0]+worldCoordsViewVec[0]*.5,ray_o1[1]+worldCoordsViewVec[1]*.5,ray_o1[2]+worldCoordsViewVec[2]*.5)
                    c=self.all_rgbs[-1][pixidx]
                    color.addData4(c[0],c[1],c[2],.6)
                    #totalidx+=1
                    pixidx+=1
            """
            
            #primitive = GeomPoints(GeomEnums.UH_static)
            #primitive.add_next_vertices(h*w)
            #geom = Geom(bodydata)
            #geom.add_primitive(primitive)
            #gnode = GeomNode('points')
            #gnode.add_geom(geom)
            #dummyRootNP.attachNewNode(gnode)
            
        #self.all_rays = torch.cat(self.all_rays,0)  # (h*w, 6) #merge all the tensors of the individual images into one giant pile.
        #print(self.all_rays)
        #exit()            
        #dummyRootNP.writeBamFile('./testfile_'+self.split+'.bam')
        #print(self.all_rgbs)
        #print(len(self.all_rgbs))
        #exit()
        #self.all_rgbs = torch.cat(self.all_rgbs, 0)
        #yeah while this is useful stuff. we can't use it as we have to work with a properly distorted real world lens.
        
        
        #self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.intrinsics[0,0],self.intrinsics[1,1]], center=self.intrinsics[:2,2])  # (h, w, 3)
        #self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        ##mind if i ask wtf? is this for outputting actual rendered images? why here?
        
        #self.center
        #self.radius
        #self.poses = []
        #self.all_rays = []
        #self.all_rgbs = []
        """
        assert len(img_files) == len(pose_files)
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), desc=f'Loading data {self.split} ({len(img_files)})'):
            image_path = os.path.join(self.root_dir, 'rgb', img_fname)
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname)) #@ self.blender2opencv
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 8)
        """
#             w2c = torch.inverse(c2w)
#
        self.img_wh = (w,h)
        self.poses = torch.stack(self.poses)
        self.all_c2w = torch.stack(self.all_c2w)
        """
        if 'train' == self.split:
            if self.is_stack:
                self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3) 
            else:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        """
        #print()
        #print(self.all_rays.shape)
        #print(self.all_rgbs.shape)
        #print(self.all_rgbs)
        
          
    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,  # Scene scaling
        factor: int = 1,                      # Image scaling (on ray gen; use gen_rays(factor) to dynamically change scale)
        scale : Optional[float] = 1.0,                    # Image scaling (on load)
        permutation: bool = True,
        white_bkgd: bool = True,
        normalize_by_bbox: bool = False,
        data_bbox_scale : float = 1,                    # Only used if normalize_by_bbox
        cam_scale_factor : float = 1,
        normalize_by_camera: bool = False,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        self.all_c2w = [] #TODO: shoult not exist, should be c2w, same for gt.
        self.all_gt = []
        self.all_intr = []
        split_name = split if split != "test_train" else "train"

        print("LOAD Meshroom DATA", root, 'split', split)

        self.split = split
        print( "splitvar",self.split)

        #def sort_key(x):
        #    if len(x) > 2 and x[1] == "_":
        #        return x[2:]
        #    return x
        #def look_for_dir(cands, required=True):
        #    for cand in cands:
        #        if path.isdir(path.join(root, cand)):
        #            return cand
        #    if required:
        #        assert False, "None of " + str(cands) + " found in data directory"
        #    return ""

        #img_dir_name = look_for_dir(["images", "image", "rgb"])
        #pose_dir_name = look_for_dir(["poses", "pose"])
        #  intrin_dir_name = look_for_dir(["intrin"], required=False)
        #img_files = sorted(os.listdir(path.join(root, img_dir_name)), key=sort_key)

        # Select subset of files
        #if self.split == "train" or self.split == "test_train":
        #    img_files = [x for x in img_files if x.startswith("0_")]
        #elif self.split == "val":
        #    img_files = [x for x in img_files if x.startswith("1_")]
        #elif self.split == "test":
        #    test_img_files = [x for x in img_files if x.startswith("2_")]
        #    if len(test_img_files) == 0:
        #        test_img_files = [x for x in img_files if x.startswith("1_")]
        #    img_files = test_img_files

        #assert len(img_files) > 0, "No matching images in directory: " + path.join(data_dir, img_dir_name)
        #self.img_files = img_files

        #dynamic_resize = scale < 1
        #self.use_integral_scaling = False
        #scaled_img_dir = ''
        #if dynamic_resize and abs((1.0 / scale) - round(1.0 / scale)) < 1e-9:
        #    resized_dir = img_dir_name + "_" + str(round(1.0 / scale))
        #    if path.exists(path.join(root, resized_dir)):
        #        img_dir_name = resized_dir
        #        dynamic_resize = False
        #        print("> Pre-resized images from", img_dir_name)
        #if dynamic_resize:
        #    print("> WARNING: Dynamically resizing images")
        self.mgdata = None #set empty default value
        self.root_dir = root
        self.__loadMainDataFile()
        self.__loadBbFromMgdata()
        self.__read_meta()
        self.c2w = self.all_c2w
        full_size = [0, 0]
        rsz_h = rsz_w = 0

      
        """
        print('NORMALIZE BY?', 'bbox' if normalize_by_bbox else 'camera' if normalize_by_camera else 'manual')
        if normalize_by_bbox:
            # Not used, but could be helpful
            bbox_path = path.join(root, "bbox.txt")
            if path.exists(bbox_path):
                bbox_data = np.loadtxt(bbox_path)
                center = (bbox_data[:3] + bbox_data[3:6]) * 0.5
                radius = (bbox_data[3:6] - bbox_data[:3]) * 0.5 * data_bbox_scale

                # Recenter
                self.c2w_f64[:, :3, 3] -= center
                # Rescale
                scene_scale = 1.0 / radius.max()
            else:
                warn('normalize_by_bbox=True but bbox.txt was not available')
        elif normalize_by_camera:
            norm_pose_files = sorted(os.listdir(path.join(root, pose_dir_name)), key=sort_key)
            norm_poses = np.stack([np.loadtxt(path.join(root, pose_dir_name, x)).reshape(-1, 4)
                                    for x in norm_pose_files], axis=0)

            # Select subset of files
            T, sscale = similarity_from_cameras(norm_poses)

            self.c2w_f64 = torch.from_numpy(T) @ self.c2w_f64
            scene_scale = cam_scale_factor * sscale

            #  center = np.mean(norm_poses[:, :3, 3], axis=0)
            #  radius = np.median(np.linalg.norm(norm_poses[:, :3, 3] - center, axis=-1))
            #  self.c2w_f64[:, :3, 3] -= center
            #  scene_scale = cam_scale_factor / radius
            #  print('good', self.c2w_f64[:2], scene_scale)
        """
        

        scene_scale = self.scene_scale
        

        print('scene_scale', scene_scale)
        
        #self.c2w_f64[:, :3, 3] *= scene_scale
        #self.c2w = self.c2w_f64.float() #we already calculated in read_meta as float.

        ##ok so we stack all images convert to mfkn double to divide by 255 and then ... oh. dear. what. no. just stick with float. drop the workarounds.
        self.gt = torch.stack(self.all_gt).double() / 255.0

        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]
        self.gt = self.gt.float()

        #assert full_size[0] > 0 and full_size[1] > 0, "Empty images"
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        print("gtstats", self.n_images, self.h_full, self.w_full )

        #intrin_path = path.join(root, "intrinsics.txt")
        #assert path.exists(intrin_path), "intrinsics unavailable"

        self.intrins_full = self.all_intr[-1]#: Intrin = Intrin(fx, fy, cx, cy)
        print(' intrinsics (loaded reso)', self.intrins_full)

        #self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full
