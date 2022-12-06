'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

from fileinput import close
from sqlite3 import SQLITE_DROP_VIEW
from turtle import width
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
import csv


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

    def getCircles(image):
        #grayscale input required
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1.4, 30, param1=700,param2=21,minRadius=0,maxRadius=50)
        for i in circles[0]:
            cv2.circle(image,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
        
        return circles[0]
    #reference camera circles
    reference_circles = getCircles(img0)
    cv2.imwrite("output/q3_circle_detection_img0.bmp", img0)
    #viewing camera circles
    viewing_circles = getCircles(img1)
    cv2.imwrite("output/q3_circle_detection_img1.bmp", img1)

    ###################################
    '''
    Question 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''

    #camera to world for reference camera
    #required for the R matrix
    H0_cw = np.linalg.inv(H0_wc)
    R = H1_wc @ H0_cw
    #remove translation component
    R2 = np.delete(np.delete(R, 3, 0), 3, 1)
    #image to pixel
    M_inv = K.intrinsic_matrix
    #pixel to image
    M = np.linalg.inv(M_inv)

    focal_length = K.get_focal_length()[0]
    ###################################

    def minimumErrorEpipolar(u, E, image_pos):
        #x * u[0] + y * u[1] = -f * u[2]
        #left side of screen
        x1 = -img_width / 2
        y1 = (-focal_length * u[2] - x1 * u[0]) / u[1]
        p1 = np.array([x1,y1,focal_length])
        xScore = p1.T @ E @ image_pos
        p1 = (p1 / focal_length) @ M_inv.T
        #right side of screen
        x2 = img_width / 2
        y2 = (-focal_length * u[2] - x2 * u[0]) / u[1]
        p2 = np.array([x2,y2,focal_length])
        x2Score = p2.T @ E @ image_pos
        p2 = (p2 / focal_length) @ M_inv.T
        #top of screen
        y1 = -img_height / 2
        x1 = (-focal_length * u[2] - y1 * u[1]) / u[0]
        py1 = np.array([x1,y1,focal_length])
        yScore = py1.T @ E @ image_pos
        py1 = (py1 / focal_length) @ M_inv.T
        #bottom of screen
        y2 = img_height / 2
        x2 = (-focal_length * u[2] - y2 * u[1]) / u[0]
        py2 = np.array([x2,y2,focal_length])
        y2Score = py2.T @ E @ image_pos
        py2 = (py2 / focal_length) @ M_inv.T
        #compare scores 
        if abs(xScore) + abs(x2Score) < abs(yScore) + abs(y2Score):
            return p1, p2, True
        return py1, py2, False

    #transformation to viewing camera relative to reference camera space
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
        p1, p2, xAxis = minimumErrorEpipolar(u, E, image_pos)
        #random colour corresponding to the epipolar line for visual identification of spheres
        colour = (random.random() * 255, random.random() * 255, random.random() * 255)
        if xAxis:
            epipolar_lines.append((p1[1],p2[1],colour,True))
        else:
            epipolar_lines.append((p1[0],p2[0],colour,False))
        #plot centers
        cv2.circle(img0, (int(i[0]), int(i[1])), 1, (0,0,0), 2)
        #outline circles with epipolar colour
        cv2.circle(img0, (int(i[0]), int(i[1])), int(i[2]), colour, 2)
        #plot line
        cv2.line(img1, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), colour, 2)
    
    cv2.imwrite("output/q4_epipolar.bmp", img1)

    ###################################
    '''
    Question 5: Find correspondences

    Write your code here
    '''
    ###################################

    matches = []
    for i in range(len(viewing_circles)):
        circle = viewing_circles[i]
        closestDistance = 10000000
        closest = -1
        #find line with least distance
        for l in range(len(epipolar_lines)):
            line = epipolar_lines[l]
            xAxis = line[3]
            if xAxis:
                t = float(circle[0]) / img_width
            else:
                t = float(circle[1]) / img_height
            #check if epipolar line has been used to force one to one mapping
            contains = False
            for match in matches:
                if match[1] == l:
                    contains = True
                    break

            if contains:
                continue

            if xAxis:
                #linearly interpolate on x to find corresponding y
                y = line[1] * t + line[0] * (1 - t)
                #distance
                dst = abs(int(y) - circle[1])
            else:
                #linearly interpolate on y to find corresponding x
                x = line[1] * t + line[0] * (1 - t)
                #distance
                dst = abs(int(x) - circle[0])
            if dst <= closestDistance:
                #new best epipolar line
                closestDistance = dst
                closest = l
                
        line = epipolar_lines[closest]
        matches.append((i, closest))
        #corresponding epipolar line colour
        col = line[2]
        cv2.circle(img1, (int(circle[0]), int(circle[1])), int(circle[2]), col, 2)
    
    cv2.imwrite("output/q5_matched.bmp", img1)
        

    ###################################
    '''
    Question 6: 3-D locations of spheres

    Write your code here
    '''
    ###################################
    #reference positions
    positions = []
    #viewing relative positions
    positions_v = []
    #ground truth index related to positions
    gt_radii = []
    gt_copy = GT_cents.copy()
    posErrors = []
    for match in matches:
        #find corresponding reference and view positions
        ref_circle = reference_circles[match[1]]
        P_Ri = np.array([ref_circle[0], ref_circle[1], 1])
        view_circle = viewing_circles[match[0]]
        P_Vi = np.array([view_circle[0], view_circle[1], 1])

        #image coordinates
        p_Ri = (M @ P_Ri)
        p_Vi = (M @ P_Vi)

        #viewing pos in relation to reference camera
        p_V_Rt = R2.T @ p_Vi

        #a * p_Ri - b * p_V_Rt - c * (p_Ri X p_V_Rt) = T
        H = np.dstack((p_Ri, p_V_Rt, np.cross(p_Ri, p_V_Rt)))
        #solve for a, b and c
        parameters = (np.linalg.inv(H) @ T).T
        #average of two lines for center position
        pos = (parameters[0] * p_Ri - parameters[1] * p_V_Rt + T) / 2

        #find closest ground truth sphere and corresponding radius
        minDst = 1000000
        minIndex = -1
        for i in range(len(gt_copy)):
            #world coordinates to h0 position
            gt = gt_copy[i]
            gt_ref = H0_wc @ gt
            dst = np.linalg.norm(np.array([pos[0], pos[1], pos[2], 1]) - gt_ref)
            if dst < minDst:
                #new closest sphere
                minIndex = i
                minDst = dst
        
        #y position in ground truth is radius as sphere sits on X Z plane
        #assumption however in application no ground truth would be given, this is only for the error value 
        minRadius = gt_copy[minIndex][1]
        gt_copy.pop(minIndex)
        #error distance between position and ground truth position
        print("Position error distance: ", minDst)
        
        posErrors.append(minDst)
        gt_radii.append(minRadius)
            
        positions.append(pos)
        pos_v = ((R @ np.array([pos[0], pos[1], pos[2], 1])))
        positions_v.append(pos_v[:3])
    
    print("Position error: ", np.mean(posErrors))
    print("Max pos error", np.max(posErrors))

    #recording average position errors
    # line = [np.mean(posErrors), np.max(posErrors)]
    # with open('pos4.csv', 'a') as file:
    #     w = csv.writer(file)
    #     w.writerow(line)
    

    ###################################
    '''
    Question 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################

    for i in range(len(matches)):
        #colour of corresponding epipolar line
        col = epipolar_lines[matches[i][1]][2]
        #corresponding estimated position
        pos = positions[i]
        #pixel position relative to reference camera
        pixel_pos = (pos / pos[2]) @ M_inv.T
        cv2.circle(img0, (int(pixel_pos[0]), int(pixel_pos[1])), 1, col, 2)
        #relative to viewing camera
        pos_v = positions_v[i]
        pixel_pos_v = (pos_v / pos_v[2]) @ M_inv.T
        cv2.circle(img1, (int(pixel_pos_v[0]), int(pixel_pos_v[1])), 1, col, 2)

    cv2.imwrite("output/q7_centers0.bmp", img0)
    cv2.imwrite("output/q7_centers1.bmp", img1)

    ###################################
    '''
    Question 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################

    #radius can be found by transforming estimated position into world space with H0_cw
    #and taking the y position as the spheres are radius above the plane
    #however I assume the task is to find the radius without world coordinates (with only coordinates related to the reference/viewing camera)

    #ground truth radius (gt_radii) is calculated in question 6 while matching the calculating corresponding ground truth spheres
    radii = []
    radiusErrors = []
    radiusErrors_r = []
    radiusErrors_v = []
    for i in range(len(matches)):
        circle_r = reference_circles[matches[i][0]]
        circle_v = viewing_circles[matches[i][0]]
        pos = positions[i]
        posDepth = pos[2]
        #depth from viewing camera perspective
        posDepth_v = (R @ np.array([pos[0], pos[1], pos[2], 1]))[2]
        #radius changes linearly with depth
        #divide by focal length because of the image plane size
        radius_r = circle_r[2] * posDepth / focal_length
        #radius by viewing camera perspective
        radius_v = circle_v[2] * posDepth_v / focal_length
        #average of both viewing for best estimate
        radius = (radius_r + radius_v) / 2
        print("Estimated radius:", radius, " | True radius: ", gt_radii[i])
        radiusErrors.append((radius - gt_radii[i]) ** 2)
        radiusErrors_r.append((radius_r - gt_radii[i]) ** 2)
        radiusErrors_v.append((radius_v - gt_radii[i]) ** 2)
        radii.append(radius)

    # comparing radius errors in different views
    line = [np.mean(radiusErrors_r), np.max(radiusErrors_r), 
        np.mean(radiusErrors_v), np.max(radiusErrors_v), 
        np.mean(radiusErrors), np.max(radiusErrors)]
    with open('radius_after.csv', 'a') as file:
        w = csv.writer(file)
        w.writerow(line)

    #recording average radius error
    # line = [np.mean(radiusErrors), np.max(radiusErrors)]
    # with open('radius4.csv', 'a') as file:
    #     w = csv.writer(file)
    #     w.writerow(line)

    ###################################
    '''
    Question 9: Display the spheres

    Write your code here:
    '''
    ###################################
    
    img0 = cv2.imread('view0.png', -1)
    img1 = cv2.imread('view1.png', -1)

    for i in range(len(matches)):
        col = epipolar_lines[matches[i][1]][2]
        pos = positions[i]
        pixel_pos = (pos / pos[2]) @ M_inv.T
        #draw the estimated circle position with the estimated radius
        #focal_length * radii[i] / pos[2] eqivilant to just using the radius from hough circles
        #just to prove that the process to find the radius earlier is reverseable and radius is linear with depth
        #and that the radius matches in both views
        cv2.circle(img0, (int(pixel_pos[0]), int(pixel_pos[1])), int(focal_length * radii[i] / pos[2]), col, 4)
        pos_v = positions_v[i]
        pixel_pos_v = (pos_v / pos_v[2]) @ M_inv.T
        cv2.circle(img1, (int(pixel_pos_v[0]), int(pixel_pos_v[1])), int(focal_length * radii[i] / pos_v[2]), col, 4)

    cv2.imwrite("output/q9_spheres0.bmp", img0)
    cv2.imwrite("output/q9_spheres1.bmp", img1)