
import cv2
import dlib
import numpy as np


#initialize dlib library face detector
# create dlib library facial landmark predictor
frontal_face_detector=dlib.get_frontal_face_detector()
frontal_face_pridector=dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")


#get the image from vedio
webcam_stream=cv2.VideoCapture(0)



#convert the source image to grayscale
sourceImage=cv2.imread("images/brucewills.jpg")
copy_of_img_s=sourceImage
sourceImage_grayscale=cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image",sourceImage_grayscale)
while True:
    #
    # destinationImage=cv2.imread("images/jason.jpg")
    destinationImage = current_frame=webcam_stream.read();
    copy_of_img_d=destinationImage
     #convert the destination image to grayscale
    destinationImage_grayscale=cv2.cvtColor(cv2.UMat(destinationImage), cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image",destinationImage_grayscale)
    
    #create zerous array canvas like same of asource image
    sourceImage_canvas=np.zeros_like(sourceImage_grayscale)
    
    #getting destination image shape
    height,width,no_of_channels=destinationImage.shape
    #create zerous array like the destination image
    destinationImage_canvas=np.zeros((height,width,no_of_channels),np.uint8)
    
    
    
    
    def index_from_array(arr):
        index=None
        for n in arr[0]:
            index=n
        return n
    
    #SOURCE IMAGE PROCESSING
    #to find the faces in the source image
    source_faces=frontal_face_detector(sourceImage_grayscale)
    #here we need to find all the landmarks of the face
    #predicator takes face as input and identify the facial landmark of it such as eyes corners,nose ..etc
    
    for source_face in source_faces:
        source_faceLandmark=frontal_face_pridector(sourceImage_grayscale,source_face)
        source_faceLandmark_points=[]
    #loop through all 68 landmark points and add them to the tuple
        for landmark_no in range(0,68):
            xpoint=source_faceLandmark.part(landmark_no).x
            ypoint=source_faceLandmark.part(landmark_no).y
            source_faceLandmark_points.append((xpoint,ypoint))
            #cv2.circle(sourceImage, (xpoint,ypoint), 2, (0,255,0))
            #cv2.imshow("",copy_of_img_s)
    #           NICE GOOD JOB LETS NOW FIND THE CONVEX HULL OF THE FACE
        source_faceLandmark_points_array=np.array(source_faceLandmark_points,np.int32)
        #find the countor from the landmark points
        sourceface_convexHull=cv2.convexHull(source_faceLandmark_points_array)
        #cv2.polylines(sourceImage,[sourceface_convexHull],True,(255,0,0),3)
        #cv2.imshow("step 2 :",sourceImage)
        #draw filled polygon
        cv2.fillConvexPoly(sourceImage_canvas,sourceface_convexHull, 255) #255 = white
        #cv2.imshow("",sourceImage_canvas)
        sourceFaceImage=cv2.bitwise_and(sourceImage,sourceImage,mask=sourceImage_canvas)
        #cv2.imshow("",sourceFaceImage)
    
        boundingRectangle=cv2.boundingRect(sourceface_convexHull)
        
        #create empty delaunay subdivition (look in wikipedia (google it ;))
        subdivitions=cv2.Subdiv2D(boundingRectangle)
        #insert the face landmark points
        subdivitions.insert(source_faceLandmark_points)
        #return triangles list as 6 numbered vectors  (3 points each with x & y)
        trianglesVectors=subdivitions.getTriangleList()
        #convert vector into numpy arr
        triangles_arr=np.array(trianglesVectors,dtype=np.int32)
        
        
        #print(triangles_arr)
        
        #we need here to draw line between each 2 points to complete the triangle
        triangle_list_source=[]
        for triangle in triangles_arr:
            edge1=(triangle[0],triangle[1])
            edge2=(triangle[2],triangle[3])
            edge3=(triangle[4],triangle[5])
            
            # edgeColor=(0,255,0) #Green
            # cv2.line(sourceFaceImage,edge1,edge2,edgeColor,1)
            # cv2.line(sourceFaceImage,edge2,edge3,edgeColor,1)
            # cv2.line(sourceFaceImage,edge3,edge1,edgeColor,1)
            # cv2.imshow("",sourceFaceImage)
            # convert the coordinate into facial landmark references
            edge1=np.where((source_faceLandmark_points_array==edge1).all(axis=1))
            edge1=index_from_array(edge1)  #index of landmark associated with eue triangle ?
            edge2=np.where((source_faceLandmark_points_array==edge2).all(axis=1))
            edge2=index_from_array(edge2)  #index of landmark associated with eue triangle ?
            edge3=np.where((source_faceLandmark_points_array==edge3).all(axis=1))
            edge3=index_from_array(edge3)  #index of landmark associated with eue triangle ?
    
            triangle=[edge1,edge2,edge3]
            triangle_list_source.append(triangle)
            
            
    #FOR DESTINATION IMAGE
    #to find the faces in the source image
    destination_faces=frontal_face_detector(destinationImage_grayscale)
    #here we need to find all the landmarks of the face
    #predicator takes face as input and identify the facial landmark of it such as eyes corners,nose ..etc
    if(len(destination_faces)==1):
    
        for destination_face in destination_faces:
            destination_faceLandmark=frontal_face_pridector(destinationImage_grayscale,destination_face)
            destination_faceLandmark_points=[]
        #loop through all 68 landmark points and add them to the tuple
            for landmark_no in range(0,68):
                xpoint=destination_faceLandmark.part(landmark_no).x
                ypoint=destination_faceLandmark.part(landmark_no).y
                destination_faceLandmark_points.append((xpoint,ypoint))
                # cv2.circle(destinationImage, (xpoint,ypoint), 2, (0,255,0))
                # cv2.imshow("",copy_of_img_d)
                
        #           NICE GOOD JOB LETS NOW FIND THE CONVEX HULL OF THE FACE
            destination_faceLandmark_points_array=np.array(destination_faceLandmark_points,np.int32)
            #find the countor from the landmark points
            destinationface_convexHull=cv2.convexHull(destination_faceLandmark_points_array)
            # cv2.polylines(destinationImage,[destinationface_convexHull],True,(255,0,0),3)
            # cv2.imshow("step 2 :",destinationImage)
            #draw filled polygon
            #cv2.fillConvexPoly(destinationImage_canvas,destinationface_convexHull, 255) #255 = white
            # cv2.imshow("",destinationImage_canvas)
               
        #for every source triangle frothe list of triangles,crop the bounding rectangle and extract only triangle points
        
        for i, triangle_indexPoint in enumerate(triangle_list_source):
            #get x and y coordinate of the source triangle 
            source_triangle_point1=source_faceLandmark_points[triangle_indexPoint[0]]
            source_triangle_point2=source_faceLandmark_points[triangle_indexPoint[1]]
            source_triangle_point3=source_faceLandmark_points[triangle_indexPoint[2]]
            
            source_triangle=np.array([ source_triangle_point1, source_triangle_point2, source_triangle_point3],np.int32)
            #draw bonding rect around triangle points
            source_rectangle=cv2.boundingRect(source_triangle)
            (x,y,w,h)=source_rectangle
            #crop source and past it in destination
            cropped_sourceRect=sourceImage[y:y+h,x:x+w]
            
            #remove  rectangle points and keep triangle points only for later use
            source_triangle_points=np.array([ [source_triangle_point1[0]-x,source_triangle_point1[1]-y],
                                              [source_triangle_point2[0]-x,source_triangle_point2[1]-y],
                                              [source_triangle_point3[0]-x,source_triangle_point3[1]-y]],np.int32)
            
            #for every destination triangle in the list of triangles
            #get x and y coordinate of the source triangle 
            destination_triangle_point1=destination_faceLandmark_points[triangle_indexPoint[0]]
            destination_triangle_point2=destination_faceLandmark_points[triangle_indexPoint[1]]
            destination_triangle_point3=destination_faceLandmark_points[triangle_indexPoint[2]]
            
            destination_triangle=np.array([ destination_triangle_point1, destination_triangle_point2, destination_triangle_point3],np.int32)
            #draw bonding rect around triangle points
            
            destination_rectangle=cv2.boundingRect(destination_triangle)
            (x,y,w,h)=destination_rectangle
            
            #crop source and past it in destination
            cropped_destinationRect=sourceImage[h,w]
            
            cropped_destinationRect_mask=np.zeros((h,w),np.uint8)
            
            #remove  rectangle points and keep triangle points only for later use
            destination_triangle_points=np.array([ [destination_triangle_point1[0]-x,destination_triangle_point1[1]-y],
                                              [destination_triangle_point2[0]-x,destination_triangle_point2[1]-y],
                                              [destination_triangle_point3[0]-x,destination_triangle_point3[1]-y]],np.int32)
        
            #put triangle points over the croped numpy zeros mask array
            cv2.fillConvexPoly(cropped_destinationRect_mask,destination_triangle_points,255)
            
            
            #converting to source triangle points
            source_triangle_points=np.float32(source_triangle_points)
            destination_triangle_points=np.float32(destination_triangle_points)
            #here we need to create the transformation matrix for warp affine method
            
            matrix=cv2.getAffineTransform(source_triangle_points,destination_triangle_points)
            
            # creating the wrapped triangle
            warped_triangle = cv2.warpAffine(cropped_sourceRect,matrix,(w,h))
            #placing the destination mask over warped triangle
            warped_triangle = cv2.bitwise_and(warped_triangle,warped_triangle, mask= cropped_destinationRect_mask)
            
            #removing the white line intriangle using mask
            New_dest_canvas_area=destinationImage_canvas[y:y+h,x:x+w]
            #convert new canvas to grayscale
            New_dest_canvas_area_gray=cv2.cvtColor(cv2.UMat(New_dest_canvas_area), cv2.COLOR_BGR2GRAY)
        
            _, mask_created_triangle=cv2.threshold(New_dest_canvas_area_gray,1,255,cv2.THRESH_BINARY_INV)
                #note it here returns two values but iam only interested in the second
            #placing the mask we have created
            wraped_triangle=cv2.bitwise_and(warped_triangle,warped_triangle,mask=mask_created_triangle)
            
            #placing masked triangle indide small canvas area
            New_dest_canvas_area=cv2.add(New_dest_canvas_area,wraped_triangle)
            #putting the triangle in the same position in destination image
            destinationImage_canvas[y:y+h,x:x+w]=New_dest_canvas_area
            cv2.imshow("11: pasting the triangle at destination canvas",destinationImage_canvas)
                # cv2.imshow("11: pasting the triangle at destination canvas",destinationImage_canvas)    
               
        # cv2.imshow("12: the completed destination canvas", destinationImage_canvas)  
        
        
        final_destination_canvas = np.zeros_like(destinationImage_grayscale)
        #cv2.imshow("13.1: the final destination canvas", final_destination_canvas)     
        
        #create the destination face mask
        final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas,destinationface_convexHull, 255)
        #cv2.imshow("13.2: the final destination face mask", final_destination_face_mask)     
            
        #invert the face mask color
        final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)    
        #cv2.imshow("13.3: the inverted final destination face mask", final_destination_face_mask)      
            
        #mask destination face
        destination_face_masked = cv2.bitwise_and(destinationImage, destinationImage, mask=final_destination_canvas)  
        #cv2.imshow("13.4: the destination_face_masked", destination_face_masked)
        
        #place new face into destination image
        destination_with_face = cv2.add(destination_face_masked,destinationImage_canvas)      
        #cv2.imshow("13.5: the destination_with_face", destination_with_face)  
            
            
            
        #Do seamless clone to make the attachment blend with the sorrounding pixels
        ###########################################################################   
        #finding the center point of the destination covex hull
        (x,y,w,h) = cv2.boundingRect(destinationface_convexHull)  
        destination_face_center_point = (int((x+x+w)/2), int((y+y+h)/2))
        
        
        #do the seamless clone   
        seamlesscloned_face = cv2.seamlessClone(destination_with_face, destinationImage, final_destination_face_mask, destination_face_center_point, cv2.NORMAL_CLONE)
        
        cv2.imshow("14: seamlesscloned_face", seamlesscloned_face)  
    else:
        cv2.imshow("14: seamlesscloned_face", destinationImage) 
        
    if(cv2.waitKey(1) & 0xff == ord('q')):
        break
#close all imshow windows when any key is pressed
# cv2.waitKey(0)
webcam_stream.release()
cv2.destroyAllWindows() 
    
    