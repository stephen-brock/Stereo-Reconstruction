'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

from fileinput import close
from turtle import width
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True

# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')
    parser.add_argument('--visual', dest='bVis', action='store_true')
    parser.add_argument('--corres', dest='corres_search', type=str, default='epipolar',
                        help='method for correspondence searching, choose from [epipolar, colour]')
    parser.add_argument('--find_sphere', dest='bDepth', action='store_true')
    args = parser.parse_args()

    img_width = 640
    img_height = 480


    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sephere with random size
        size = random.randrange(10, 14, 2) / 10.
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # random set sphere location
        step = 6
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))
        print(f'sphere_{i}: [{size}, {x}, {z}] {size}')
        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    # set camera intrinsics
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 415.69219381653056, 415.69219381653056, 319.5, 239.5)
    print('Camera parameters:')
    print('Pose_0\n', H0_wc)
    print('Pose_1\n', H1_wc)
    print('Intrinsics\n', K.intrinsic_matrix)
    # o3d.io.write_pinhole_camera_intrinsic("test.json", K)


    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    
    ###################################
    '''
    Question 3: Circle detection
    Hint: check cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################

    def getCircles(image, image_name):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1.5, 30, param1=600,param2=20,minRadius=0,maxRadius=50)
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
        
        # cv2.imwrite("output/" + image_name + "_circles.bmp", image)
        return circles[0]
    reference_circles = getCircles(img0, "image0")
    viewing_circles = getCircles(img1, "image1")

    ###################################
    '''
    Question 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################

    #camera to world for reference camera
    H0_cw = np.linalg.inv(H0_wc)
    M_inv = K.intrinsic_matrix
    M = np.linalg.inv(M_inv)
    focal_length = K.get_focal_length()[0]
    #transformation to viewing camera relative to reference camera space
    R = H1_wc @ H0_cw
    #remove translation component
    R2 = np.delete(np.delete(R, 3, 0), 3, 1)
    #viewing position relative to reference
    T = np.linalg.inv(R) @ np.array([0,0,0,1])
    T = np.delete(T, 3, 0)
    epipolar_lines = []
    for i in reference_circles:
        #circle center in reference camera pixel coordinates
        pixel_pos = np.array([i[0], i[1], 1])
        #reference camera image space
        image_pos = M @ pixel_pos
        #cross product matrix
        S = np.array([
            [0,-T[2], T[1]], 
            [T[2], 0, -T[0]], 
            [-T[1], T[0], 0]])
        #essential matrix
        E = R2 @ S

        #to solve for x/y
        u = E @ image_pos
        #x * u[0] + y * u[1] = -f * u[2]
        #left side of screen
        x1 = -img_width / 2
        y1 = (-focal_length * u[2] - x1 * u[0]) / u[1]
        p1 = (np.array([x1,y1,focal_length]) / focal_length) @ M_inv.T
        #right side of screen
        x2 = img_width / 2
        y2 = (-focal_length * u[2] - x2 * u[0]) / u[1]
        p2 = (np.array([x2,y2,focal_length]) / focal_length) @ M_inv.T
        colour = (random.random() * 255, random.random() * 255, random.random() * 255)
        epipolar_lines.append((p1[1],p2[1],colour))
        #plot line
        cv2.circle(img0, (i[0], i[1]), 2, (0,0,0), 4)
        cv2.circle(img0, (i[0], i[1]), i[2], colour, 2)
        cv2.line(img1, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), colour, 2)
    
    cv2.imwrite("output/img0_epipolar.bmp", img0)
    cv2.imwrite("output/img1_epipolar.bmp", img1)

    ###################################
    '''
    Question 5: Find correspondences

    Write your code here
    '''
    ###################################

    epipolar_lines_copied = epipolar_lines.copy()
    matches = []
    for i in range(len(viewing_circles)):
        circle = viewing_circles[i]
        closestDistance = 10000000
        closest = -1
        t = float(circle[0]) / img_width
        for l in range(len(epipolar_lines_copied)):
            line = epipolar_lines_copied[l]
            y = line[1] * t + line[0] * (1 - t)
            dst = abs(int(y) - circle[1])
            if dst <= closestDistance:
                closestDistance = dst
                closest = l
                
        line = epipolar_lines_copied[closest]
        y = line[1] * t + line[0] * (1 - t)
        matches.append((i, closest))
        cv2.circle(img1, (circle[0], circle[1]), circle[2], line[2],2)
    
    cv2.imwrite("output/matched_circles.bmp", img1)
        

    ###################################
    '''
    Question 6: 3-D locations of spheres

    Write your code here
    '''
    ###################################
    positions = []
    for match in matches:
        ref_circle = reference_circles[match[1]]
        P_Ri = np.array([ref_circle[0], ref_circle[1], 1])
        view_circle = viewing_circles[match[0]]
        P_Vi = np.array([view_circle[0], view_circle[1], 1])

        p_Ri = (M @ P_Ri)
        p_Vi = (M @ P_Vi)

        p_V_Rt = R2.T @ p_Vi
        #a * p_Ri - b * p_V_Rt - c * (p_Ri X p_V_Rt) = T
        H = np.dstack((p_Ri, p_V_Rt, np.cross(p_Ri, p_V_Rt)))
        parameters = (np.linalg.inv(H) @ T).T
        pos = (parameters[0] * p_Ri - parameters[1] * p_V_Rt + T) / 2
        positions.append(pos)
        pos /= pos[2]
        pixel_pos = pos @ M_inv.T
        col = epipolar_lines[match[1]][2]
        cv2.circle(img0, (int(pixel_pos[0]), int(pixel_pos[1])), 2, col, 4)
    
    cv2.imwrite("output/3d-circles.bmp", img0)

    ###################################
    '''
    Question 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Question 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Question 9: Display the spheres

    Write your code here:
    '''
    ###################################
