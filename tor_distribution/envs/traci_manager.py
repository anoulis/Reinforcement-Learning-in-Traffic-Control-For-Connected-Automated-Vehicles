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
## Network configuration
# This script assumes that there is a single edge ('e0') on which the ToCs are taking place.
# Entry position of the NoAD zone on edge e0
NOAD_ZONE_ENTRY_POS = 2300.

class Vehicle:
    def __init__(self, pos, speed, lane, length, minGap, vehID, automationType, detectionTime, cell, wt):
        self.pos=pos
        self.speed=speed
        self.lane=lane
        self.minGap = minGap
        self.length = length
        self.ID=vehID
        self.TORstate='noTOR'
        self.automationType=automationType
        self.detectionTime = detectionTime
        self.origColor = traci.vehicle.getColor(vehID)
        self.cell = cell
        self.wt = wt
    def setState(self,TOR):
        self.TORstate=TOR
    def updateCell(self,cell):
        self.cell = cell
    def updateWT(self, wt):
        self.wt = wt
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
    def getCell(self):
        return self.cell



# The old runner code
class TraciManager():

    def __init__(self,cells_number):
        self.MAX_OCCUPANCY = 10.

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
        assert(self.areaLengths == [NOAD_ZONE_ENTRY_POS - self.areaPos[0]]*2)
        # Platoons being currently formed (unclosed) on the two lanes ("trailing platoons")
        self.openPlatoons = [None for _ in self.loops]
        # Last step detected vehicles (to avoid duplicate additions to platoons)
        self.lastStepDetections = [set() for _ in self.loops]

        self.activatedCell = self.zerolistmaker(2)

        self.latePunishment = 0

        # List of CAV/CVs in ToC zone.
        self.CAV_CV = []
        # List of LVs in ToC zone
        self.LVsInToCZone = []
        # List of C(A)Vs, which received a TOR
        self.pendingToCVehs = []

        # Lists for storing important values for every cell
        self.vehsAVPerCell = []
        self.vehsPendPerCell = []
        self.vehsLVPerCell = []
        self.speedPerCell = []
        self.densityPerCell = []
        self.WTPerCell = []

        # Varius useful values
        self.step = 0
        self.sendToCs = self.zerolistmaker(2)
        self.totalSendToCs = 0
        self.cells_number = cells_number
        self.ToC_Per_Cell = self.zerolistmaker(self.cells_number)
        self.cav_dist = 0
        self.forcedToCs = 0
        self.missed = []

    def requestToC(self, vehID, vehCell, vehPos, timeUntilMRM):
        """ 
        The functiion that implements the ToC request through traci.
        """
        self.ToC_Per_Cell[vehCell-1]+=1
        # it just sum ups the distnace that have covered by ca/cav veh
        self.cav_dist += vehPos
        traci.vehicle.setParameter(vehID, "device.toc.requestToC", str(timeUntilMRM))

    def removeVehiclesBeyond(self, x, vehList):
        ''' removeVehiclesBeyond(float, vehList)
        Removes all vehicles with position > x from the list.
        '''
        toRemove=[veh for veh in vehList if veh.pos > x]
        for veh in toRemove:
            vehList.remove(veh)

    def getCell(self,pos,lane):
        """
        Return the cell that a veh belongs, base the position
        """
        if(self.cells_number==10):
            if lane == "e0_0":
                if pos<350:
                    return 2
                elif pos>350 and pos<850:
                    return 4
                elif pos>850 and pos<1400:
                    return 6
                elif pos>1400 and pos<2000:
                    return 8
                else:
                    return 10
            else:
                if pos<350:
                    return 1
                elif pos>350 and pos<850:
                    return 3
                elif pos>800 and pos<1400:
                    return 5
                elif pos>1400 and pos<2000:
                    return 7
                else:
                    return 9
        else:
            if lane == "e0_0":
                if pos < 300:
                    return 2
                elif pos > 300 and pos < 650:
                    return 4
                elif pos > 650 and pos < 1000:
                    return 6
                elif pos > 1000 and pos < 1350:
                    return 8
                elif pos > 1350 and pos < 1700:
                    return 10
                elif pos > 1700 and pos < 2000:
                    return 12
                else:
                    return 14
            else:
                if pos < 300:
                    return 1
                elif pos > 300 and pos < 650:
                    return 3
                elif pos > 650 and pos < 1000:
                    return 5
                elif pos > 1000 and pos < 1350:
                    return 7
                elif pos > 1350 and pos < 1700:
                    return 9
                elif pos > 1700 and pos < 2000:
                    return 11
                else:
                    return 13

    def countVehsPerCells(self,vehList):
        """
        Returns a list with the number of specific veh.
        CAV_CV, pendingToCVehs, LVsInToCZone
        """
        storeList = self.zerolistmaker(self.cells_number)
        for veh in vehList:
            storeList[veh.cell-1] += 1
        return storeList

    def setSpeedperCells(self):
        """
        Store the Average speed per cell
        """
        sum = self.zerolistmaker(self.cells_number)
        count = self.zerolistmaker(self.cells_number)
        self.speedPerCell = self.zerolistmaker(self.cells_number)
        for veh in self.CAV_CV:
            sum[veh.cell-1] += veh.speed
            count[veh.cell-1] += 1
        for veh in self.pendingToCVehs:
            sum[veh.cell-1] += veh.speed
            count[veh.cell-1] += 1
        for veh in self.LVsInToCZone:
            sum[veh.cell-1] += veh.speed
            count[veh.cell-1] += 1
        for i in range(self.cells_number):
            if(count[i]==0):
                self.speedPerCell[i] = 0
            else:
                self.speedPerCell[i] += sum[i]/count[i]

    def setWTperCells(self):
        """
        Store the Total Accumulated WT per cell
        """
        self.WTPerCell = self.zerolistmaker(self.cells_number)
        for veh in self.CAV_CV:
            self.WTPerCell[veh.cell-1] += veh.wt
        for veh in self.pendingToCVehs:
            self.WTPerCell[veh.cell-1] += veh.wt
        for veh in self.LVsInToCZone:
            self.WTPerCell[veh.cell-1] += veh.wt

    def setdensityperCells(self):
        """
        Store the density of vehs per cell
        """
        sum = 0
        self.densityPerCell = self.zerolistmaker(self.cells_number)
        for i in range(self.cells_number):
            sum = self.vehsAVPerCell[i] + self.vehsPendPerCell[i] + self.vehsLVPerCell[i]
            if(i==1 or 1==2):
                self.densityPerCell[i]= sum/300.
            else:
                self.densityPerCell[i]= sum/500.

    def getDensityPerCells(self):
        """ Returns the list with densities for all cells """
        return self.densityPerCell

    def getWTPerCells(self):
        """ Returns the ACC_WT for all cells """
        return self.WTPerCell

    def getStep(self):
        """ Returns the Simulation step"""
        return self.step

    def getSpeedPerCells(self):
        """ Returns the list with Average speed for all cells """
        return self.speedPerCell

    def getAVperCells(self):
        """ Returns the list with the number of CAV_CV for all cells """
        return self.vehsAVPerCell

    def getPendperCells(self):
        """ Returns the list with the number of pendingToCVehs for all cells """
        return self.vehsPendPerCell

    def getLVperCells(self):
        """ Returns the list with the number of LVsInToCZone for all cells """
        return self.vehsLVPerCell

    def getDecidedToCs(self):
        """ Returns the number of the sended ToC messages """
        return self.sendToCs

    def getCellInfluence(self,cell):
        """ Returns the influence of each cell on the reward"""
        if(self.cells_number == 10):
            if cell==2 or cell==1:
                return 0.1
            elif cell==4 or cell==3:
                return 0.2
            elif cell==6 or cell==5:
                return 0.3
            elif cell==8 or cell==7:
                return 0.4
            else:
                # return o.5
                return 0
        else:
            if cell == 2 or cell == 1:
                return 0.1
            elif cell == 4 or cell == 3:
                return 0.2
            elif cell == 6 or cell == 5:
                return 0.3
            elif cell == 8 or cell == 7:
                return 0.4
            elif cell == 10 or cell == 9:
                return 0.5
            elif cell == 12 or cell == 11:
                return 0.6
            else:
                return 0

    def zerolistmaker(self,n):
        """ Creates a list of zeros"""
        listofzeros = [0] * n
        return listofzeros

    def getRatio(self):
        if(len(self.CAV_CV)!=0):
            return len(self.pendingToCVehs)/len(self.CAV_CV)
        else:
            return  len(self.pendingToCVehs)+1

    # Various get functions
    def getAreaDet(self):
        """ Returns the area Detectos """
        return self.areaDetectors

    def getTrafficJams(self):
        """ Retuns sum of trafficJams for the 2 lanes"""
        tj= 0
        dets = self.areaDetectors
        for i in range(2):
            tj += traci.lanearea.getJamLengthMeters(dets[i])
        return tj

    # def set_late_punishment(self):
    #     """
    #     Old way for calculation of late forced ToCs.

    #     We count the forced ToC to vehs that have approached the end
    #     of the zone and didn't receive ToC message.
    #     We store that message, so as to get penalized in rewards calculations.
    #     """
    #     punishment = 0
    #     if(self.cells_number == 10):
    #         limit =2000
    #     else:
    #         limit = 2200
    #     for veh in self.CAV_CV:
    #         if(veh not in self.pendingToCVehs):
    #             if(traci.vehicle.getDistance(veh.ID)>limit):
    #                 self.requestToC(veh.ID, veh.cell, veh.pos, ToC_lead_times[veh.automationType])
    #                 self.pendingToCVehs.append(veh)
    #                 self.CAV_CV.remove(veh)
    #                 punishment+=1
    #     self.latePunishment = punishment
    #     self.forcedToCs += self.latePunishment

    def sendForced(self):
        """
        New way for sending forced ToCs

        We send the forced ToC to vehs that have approached the end
        of the zone and didn't receive ToC message.
        """
        limit = 2000
        for veh in self.missed:
            if(veh not in self.pendingToCVehs):
                if(traci.vehicle.getDistance(veh.ID) > limit and (veh.cell == 13 or veh.cell == 14)):
                    self.requestToC(veh.ID, veh.cell, veh.pos,
                                    ToC_lead_times[veh.automationType])
                    self.pendingToCVehs.append(veh)
                    self.CAV_CV.remove(veh)
                    self.missed.remove(veh)


    def get_forced_ToCs(self):
        """ Retuns sum of forced ToC"""
        return self.latePunishment

    def getAreaLanes(self):
        """ Retuns the list of areaLanes"""
        return self.areaLanes

    def getIDList(self):
        """ Retuns list with all the veh ids"""
        return traci.vehicle.getIDList()

    def getVehNum(self):
        """ Retuns the sum of vehs"""
        return traci.vehicle.getIDCount()

    def getLanes(self):
        """Returns lanes ids"""
        list = traci.lane.getIDList()
        lanes =[]
        for item in list:
            lanes.append(traci.lane.getEdgeID(item))
        return lanes

    def getLaneVehNum(self,det):
        """ Returns the sum of vehs for specific lane"""
        return traci.lanearea.getLastStepVehicleNumber(det)

    def getLaneWait(self,lane):
        """ Returns the waiting time for specific lane"""
        return traci.lane.getWaitingTime(lane)

    def getLaneMeanSpeed(self,lane):
        """ Returns the mean speed of vehs for specific lane"""
        return traci.lane.getLastStepMeanSpeed(lane)

    def getLaneTravelTime(self,lane):
        """ Returns the mean speed of vehs for specific lane"""
        return traci.lane.getTraveltime(lane)

    def getVehWaitTime(self,veh):
        """ Returns the waitingTime for specific veh"""
        return traci.vehicle.getAccumulatedWaitingTime(veh)

    def sim_step(self):
        """ Returns the simulation time"""
        return traci.simulation.getTime()

    def updatePositions(self, vehicleList):
        ''' updatePositions() -> None
        Updates the lanes and positions for the given lists of vehicles and platoons.
        '''
        # print("updatePositions() for vehicles: %s"%str([veh.ID for veh in vehicleList]))
        for veh in vehicleList:
            traciResults = traci.vehicle.getSubscriptionResults(veh.ID)
            #print("traciResults: %s"%str(traciResults))
            veh.pos = traciResults[tc.VAR_LANEPOSITION]
            veh.speed = traciResults[tc.VAR_SPEED]
            veh.lane = traciResults[tc.VAR_LANE_ID]

    def subscribeState(self,vehID):
        ''' subscribeState(string) -> None
        Adds a traci subscription for the vehicle's state (lanePos, speed, laneID)
        '''
        # print("subscribeState() for vehicle '%s'"%vehID)
        traci.vehicle.subscribe(vehID, [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED])

    def do_steps(self,steps):
        if(steps>0):
            self.activatedCell = self.zerolistmaker(2)
            for i in range(steps):
                i +=1
                self.call_runner()

    def call_runner(self):
        """ Main execution function of the sumo simulation"""

        self.step +=1
        self.sendToCs = self.zerolistmaker(2)
        self.totalSendToCs = 0
        # self.ToCs = 0
        traci.simulationStep()
        t = traci.simulation.getTime()
        # print("\n---------------------------------\nstep: %s (time=%s)"%(step, t))
        # print("---------------------------------")

        self.removeVehiclesBeyond(NOAD_ZONE_ENTRY_POS, self.LVsInToCZone)
        self.removeVehiclesBeyond(NOAD_ZONE_ENTRY_POS, self.pendingToCVehs)
        self.removeVehiclesBeyond(NOAD_ZONE_ENTRY_POS, self.CAV_CV)

        self.updatePositions(self.LVsInToCZone)
        self.updatePositions(self.pendingToCVehs)
        self.updatePositions(self.CAV_CV)

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
                self.subscribeState(vehID)

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
                                  ID, self.automationType, t, self.getCell(pos,self.loopLane),0)

                    if self.automationType != 'LV.':
                        # Last entered vehicle is automated
                        assert(self.automationType=="CAVToC." or self.automationType=="CVToC.")
                        # temporaly commented as it stops simultation
                        # if (self.loopLane != traci.vehicle.getLaneID(ID)):
                        #     warn("The detected vehicle's lane differs from entry loop's lane! Please assure that the loop is far enough from the lane end to prevent this situation.")
                    #                         raise Exception("The detected vehicle's lane differs from entry loop's lane! Please assure that the loop is far enough from the lane end to prevent this situation.")
                        self.CAV_CV.append(veh)
                    else:
                        # Last entered vehicle is an LV, close current platoon
                        assert(self.automationType=="LV.")
                        self.LVsInToCZone.append(veh)

        self.lastStepDetections = set()
        for vehs in self.detected:
            self.lastStepDetections.update(vehs)

        for veh in self.pendingToCVehs:
            veh.updateCell(self.getCell(veh.pos,veh.lane))
            veh.updateWT(traci.vehicle.getAccumulatedWaitingTime(veh.ID))

        for veh in self.LVsInToCZone:
            veh.updateCell(self.getCell(veh.pos,veh.lane))
            veh.updateWT(traci.vehicle.getAccumulatedWaitingTime(veh.ID))

        for veh in self.CAV_CV:
            veh.updateCell(self.getCell(veh.pos, veh.lane))
            if(veh.cell == 13 or veh.cell==14):
                if( veh not in self.missed):
                    self.missed.append(veh)
            veh.updateWT(traci.vehicle.getAccumulatedWaitingTime(veh.ID))
            if(veh not in self.pendingToCVehs):
                # if(self.activatedCell!=0):
                # print()
                if (veh.cell == self.activatedCell[0] or veh.cell == self.activatedCell[1]):
                    # self.calculate_early_punishment(veh.cell)
                    self.requestToC(veh.ID, veh.cell, veh.pos, ToC_lead_times[veh.automationType])
                    self.totalSendToCs =+1
                    if(veh.cell >6):
                        self.sendToCs[1] = 1
                    else:
                        self.sendToCs[0] = 1
                    self.pendingToCVehs.append(veh)
                    self.CAV_CV.remove(veh)

        self.vehsAVPerCell = self.countVehsPerCells(self.CAV_CV)
        self.vehsPendPerCell = self.countVehsPerCells(self.pendingToCVehs)
        self.vehsLVPerCell = self.countVehsPerCells(self.LVsInToCZone)
        self.setSpeedperCells()
        self.setdensityperCells()
        self.setWTperCells()
        # Dropped approach for calcuation late send ToCs
        # self.set_late_punishment()
