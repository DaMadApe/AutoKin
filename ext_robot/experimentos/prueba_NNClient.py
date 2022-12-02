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
    # if not frameNumber%100:
    #     print("Frame", frameNumber )
    #     print("markerSetCount", markerSetCount)
    #     print("unlabeledMarkersCount", unlabeledMarkersCount)
    #     print("rigidBodyCount", rigidBodyCount)
    #     print("skeletonCount", skeletonCount)
    #     print("labeledMarkerCount", labeledMarkerCount)
    #     print("timecode", timecode)
    #     print("timecodeSub", timecodeSub)
    #     print("timestamp", timestamp)
    #     print("isRecording", isRecording)
    #     print("trackedModelsChanged", trackedModelsChanged)

    # time.sleep(0.5)
    pass

def recv_body(id, position, rotation):
    # called once per rigid body per frame
    body_pos = None
    if id == 1: # Base del robot
        body_pos = position
        # xb, yb, zb = position
        x, y, z = position
        print(f"body1: x: {x:.6f}, y: {y:.6f}, z: {z:.6f}", 
              end='\r')
    if id == 2: # and body_pos is not None:
        xr, yr, zr = position
        print(f"xr: {xr:.6f}, yr: {yr:.6f}, zr: {zr:.6f}", 
            end='\r')

    # x, y, z = position
    # print(f"body{id}: x: {x:.6f}, y: {y:.6f}, z: {z:.6f}")
        # print(f"Id: {id}, pos: {position:.4f}, rot: {rotation}", end='\r')
    # time.sleep(0.5)


client = NatNetClient()
client.newFrameListener = recv_frame
client.rigidBodyListener = recv_body
client.run()

print('running')

input()

# print('Units in mm', send_cam_command(client, 'UnitsToMillimeters'))
# print( 'FrameRate', send_cam_command(client, 'FrameRate'))