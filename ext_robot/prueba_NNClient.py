from ext_robot.NatNetClient import NatNetClient


def recv_frame(self, frameNumber, 
                markerSetCount,unlabeledMarkersCount,
                rigidBodyCount,skeletonCount,labeledMarkerCount,
                timecode,timecodeSub,timestamp,
                isRecording,trackedModelsChanged):
    print( "Frame", frameNumber )

def recv_body(self, id, position, rotation):
    # called once per rigid body per frame
    print(f"Id: {id}, pos: {position}, rot: {rotation}")


client = NatNetClient()
client.newFrameListener = recv_frame
client.rigidBodyListener = recv_body
client.run()