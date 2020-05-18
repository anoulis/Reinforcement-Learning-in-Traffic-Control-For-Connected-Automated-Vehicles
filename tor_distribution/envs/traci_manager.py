import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import traci.constants as tc
from scenario.example import runner

def getIdentifier(fullID, identifierList):
    #~ print ("getIdentifier(%s, %s)"%(fullID, identifierList))
    for start_str in identifierList:
        if fullID.startswith(start_str):
            #~ print("found %s"%start_str)
            return start_str

ToC_lead_times = {"CAVToC.":10.0, "CVToC.":0.0, "LV.":-1.0}

def requestToC(vehID, timeUntilMRM):
    traci.vehicle.setParameter(vehID, "device.toc.requestToC", str(timeUntilMRM))

class Vehicle:
    def __init__(self, pos, speed, lane, length, minGap, vehID, automationType, detectionTime):
        self.pos=pos
        self.speed=speed
        self.lane=lane
        self.minGap = minGap
        self.length = length
        self.ID=vehID
        self.TORstate='noTOR'
        self.automationType=automationType
        self.detectionTime = detectionTime
        self.xTOR = None
        self.thTOR = None
        self.origColor = traci.vehicle.getColor(vehID)
    def setState(self,TOR):
        self.TORstate=TOR
    def updatePosition(self):
        self.pos=traci.vehicle.getLanePosition(self.ID)
    def getID(self):
        return self.ID
    def getAutomationType(self):
        return self.automationType
    def getLane(self):
        return self.lane
    def getPos(self):
        return self.pos
    def getState(self):
        return self.TORstate


# The old runner code
class TraciManager():

    def __init__(self):
        self.downwardEdgeID = "e0"
        self.distance = 2300.
        """execute the TraCI control loop
        """
        global NOAD_ZONE_ENTRY_POS
        # Induction loop IDs for entry detectors
        self.loops = ['loop0', 'loop1']
        self.loopPos = [traci.inductionloop.getPosition(loopID) for loopID in self.loops]
        self.loopLanes = [traci.inductionloop.getLaneID(loopID) for loopID in self.loops]
        # area detectors for densities
        self.areaDetectors = ['area0', 'area1']
        self.areaPos = [traci.lanearea.getPosition(areaID) for areaID in self.areaDetectors]
        self.areaLanes = [traci.lanearea.getLaneID(areaID) for areaID in self.areaDetectors]
        self.areaLengths = [traci.lanearea.getLength(areaID) for areaID in self.areaDetectors]
        assert(self.areaPos == self.loopPos)
        assert(self.areaLanes == self.loopLanes)
        assert(self.areaLengths == [runner.NOAD_ZONE_ENTRY_POS - self.areaPos[0]]*2)
        # Platoons being currently formed (unclosed) on the two lanes ("trailing platoons")
        self.openPlatoons = [None for _ in self.loops]
        # Last step detected vehicles (to avoid duplicate additions to platoons)
        self.lastStepDetections = [set() for _ in self.loops]

        # List of CAV/CV platoons in ToC zone. Map: platoon-ID -> platoon object
        self.CAV_CV = []
        # List of LVs in ToC zone
        self.LVsInToCZone = []
        # List of C(A)Vs, which received a TOR
        self.pendingToCVehs = []
        self.early_ToC = 0


    # Various get functions
    def getAreaDet(self):
        return self.areaDetectors

    def get_late_punishment(self):
        punishment = 0
        for veh in self.CAV_CV:
            if(veh not in self.pendingToCVehs):
                if(traci.vehicle.getDistance(veh.ID)>2000):
                    requestToC(veh.ID, ToC_lead_times[veh.automationType])
                    self.pendingToCVehs.append(veh)
                    punishment+=5
        return punishment

    def calculate_early_punishment(self,veh_ID):
        if(traci.vehicle.getDistance(veh_ID)<1500):
            self.early_ToC+=4

    def get_early_punishment(self):
        return self.early_ToC

    def getAreaLanes(self):
        return self.areaLanes

    def getIDList(self):
        return traci.vehicle.getIDList()

    def getVehNum(self):
        return traci.vehicle.getIDCount()

    def getLanes(self):
        list = traci.lane.getIDList()
        lanes =[]
        for item in list:
            lanes.append(traci.lane.getEdgeID(item))
        return lanes

    def getLaneVehNum(self,det):
        return traci.lanearea.getLastStepVehicleNumber(det)

    def getLaneWait(self,lane):
        return traci.lane.getWaitingTime(lane)

    def getLaneMeanSpeed(self,lane):
        return traci.lane.getLastStepMeanSpeed(lane)

    def getVehWaitTime(self,veh):
        return traci.vehicle.getAccumulatedWaitingTime(veh)

    def sim_step(self):
        return traci.simulation.getTime()

    def call_runner(self,step,flag):
        traci.simulationStep()
        t = traci.simulation.getTime()
        # print("\n---------------------------------\nstep: %s (time=%s)"%(step, t))
        # print("---------------------------------")
        runner.removeVehiclesBeyond(runner.NOAD_ZONE_ENTRY_POS, self.LVsInToCZone)
        runner.removeVehiclesBeyond(runner.NOAD_ZONE_ENTRY_POS, self.pendingToCVehs)

        self.detected = [traci.inductionloop.getLastStepVehicleIDs(loopID) for loopID in self.loops]
        self.occupancyLevels = [min(1.0, traci.lanearea.getLastStepOccupancy(areaID)/runner.MAX_OCCUPANCY) for areaID in self.areaDetectors]


        for idx in [0,1]:
            self.loopLane = self.loopLanes[idx]
            # new vehs on this lane
            self.ids = [ID for ID in self.detected[idx] if not ID in self.lastStepDetections]
            self.positions = [traci.vehicle.getLanePosition(ID) for ID in self.ids]
            # print("Entered vehicles at lane %s: %s"%(idx, str(self.ids)))
            # print("Occupancy of lane %s: %s"%(idx, self.occupancyLevels[idx]))
            # Subscribe to automatic state updates for new vehicle
            for vehID in self.ids:
                runner.subscribeState(vehID)

                # List of pairs (id, pos), sorted descendingly in pos
                self.sortedVehPositions = [(i,p) for (p,i) in reversed(sorted(zip(self.positions, self.ids)))]

                # Manage detected vehicles.
                # - Add C(A)Vs to CAV_CV
                # - Add LVs to LVsInToCZone
                for (ID, pos) in self.sortedVehPositions:
                    self.automationType = getIdentifier(ID, ToC_lead_times.keys())
                    # Create Vehicle object
                    veh = Vehicle(pos, traci.vehicle.getSpeed(ID), self.loopLane, \
                                  traci.vehicle.getLength(ID), traci.vehicle.getMinGap(ID), \
                                  ID, self.automationType, t)

                    if self.automationType != 'LV.':
                        # Last entered vehicle is automated
                        assert(self.automationType=="CAVToC." or self.automationType=="CVToC.")
                        if (self.loopLane != traci.vehicle.getLaneID(ID)):
                            warn("The detected vehicle's lane differs from entry loop's lane! Please assure that the loop is far enough from the lane end to prevent this situation.")
                    #                         raise Exception("The detected vehicle's lane differs from entry loop's lane! Please assure that the loop is far enough from the lane end to prevent this situation.")
                        self.CAV_CV.append(veh)
                    else:
                        # Last entered vehicle is an LV, close current platoon
                        assert(self.automationType=="LV.")
                        self.LVsInToCZone.append(veh)

        self.lastStepDetections = set()
        for vehs in self.detected:
            self.lastStepDetections.update(vehs)

        self.early_ToC = 0
        for veh in self.CAV_CV:
            # print(str(veh.ID) + "  " + str(traci.vehicle.getSpeed(veh.ID))
            # + "  " + str(traci.vehicle.getLateralSpeed(veh.ID))
            # + "  " + str(traci.vehicle.getAcceleration(veh.ID))
            # + "  " + str(traci.vehicle.getTypeID(veh.ID)))
            if(veh not in self.pendingToCVehs):
                if(flag):
                    self.calculate_early_punishment(veh.ID)
                    requestToC(veh.ID, ToC_lead_times[veh.automationType])
                    self.pendingToCVehs.append(veh)
        self.get_late_punishment()
