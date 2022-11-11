import time

from ext_robot.NatNetClient import NatNetClient

def send_cam_command(cam_client: NatNetClient, command_str):
    cam_client.sendCommand(command=cam_client.NAT_REQUEST,
                           commandStr=command_str,
                           socket=cam_client.commandSocket,
                           address=(cam_client.serverIPAddress,
                                    cam_client.commandPort))

def recv_frame(frameNumber, 
               markerSetCount,unlabeledMarkersCount,
               rigidBodyCount,skeletonCount,labeledMarkerCount,
               timecode,timecodeSub,timestamp,
               isRecording,trackedModelsChanged):
    # print("Frame", frameNumber )
    # print("markerSetCount", markerSetCount)
    # print("unlabeledMarkersCount", unlabeledMarkersCount)
    # print("rigidBodyCount", rigidBodyCount)
    # print("skeletonCount", skeletonCount)
    # print("labeledMarkerCount", labeledMarkerCount)
    # print("timecode", timecode)
    # print("timecodeSub", timecodeSub)
    # print("timestamp", timestamp)
    # print("isRecording", isRecording)
    # print("trackedModelsChanged", trackedModelsChanged)

    # time.sleep(0.5)
    pass

def recv_body(id, position, rotation):
    # called once per rigid body per frame
    if id == 2:
        x, y, z = position
        r, i, j, k = rotation
        print(f"x: {x:.6f}, y: {y:.6f}, z: {z:.6f}, r: {r:.6f}, i: {i:.6f}, j: {j:.6f}, k: {k:.6f}", 
              end='\r')
        # print(f"Id: {id}, pos: {position:.4f}, rot: {rotation}", end='\r')
    # time.sleep(0.5)


client = NatNetClient()
client.newFrameListener = recv_frame
client.rigidBodyListener = recv_body
client.run()

print('running')


# print('Units in mm', send_cam_command(client, 'UnitsToMillimeters'))
# print( 'FrameRate', send_cam_command(client, 'FrameRate'))