import pygame, sys
import math
from random import randint
import random
from random import seed
import numpy as np

seed(12345)
pygame.init()
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
BUTTON = (180, 180, 180)
GREY   = (127, 127, 127)
BACK   = (227, 227, 227)
BLACK  = (0,0,0)
WHITE  = (255, 255, 255)
BLUE   = (0, 0, 255)
mouse_x,mouse_y = 0,0
FPS = 30
FramePerSec = pygame.time.Clock()
normalImg = pygame.image.load('personOther.png')
yourselfImg = pygame.image.load('personYou.png')
bestImg = pygame.image.load('person.png')
deadImg = pygame.image.load('personDead.png')
icon   = pygame.image.load('icon.png')
DISPLAYSURF = pygame.display.set_mode((1200,750))
pygame.display.set_caption('ai learns to walk')
pygame.display.set_icon(icon)
clk,gen = 0,0
isPressingLeft = False
isPressingRight = False
scores = [0]*21
rects = [[0,0,25,1205],
         [1180,0,25,1205],
         [25,-10,1205,21],
         [25,740,1205,25],
         [175,160,855,430]]
lines = [[(23.0, 10.0), (23.0, 740.0)],
         [(1180.0, 10.0), (1180.0, 740.0)],
         [(23.0, 8.0), (1180.0, 8.0)],
         [(23.0, 740.0), (1180.0, 740.0)],
         [(175.0, 160.0), (175.0, 588.0)],
         [(175.0, 160.0), (1028.0, 160.0)],
         [(1028.0, 162.0), (1028.0, 590.0)],
         [(175.0, 588.0), (1028.0, 588.0)]]

def colision(circleX, circleY, circleR, rectX, rectY, rectW, rectH):
    if(circleX+circleR > rectX and circleY+circleR > rectY and
       circleX-circleR < rectX+rectW and circleY-circleR < rectY+rectH):
        return True
    else:
        return False

def GetDistance(x1, y1, x2, y2):
    return math.sqrt(((x2-x1)**2) + ((y2-y1)**2))

def slope(p1, p2) :
   return (p2[1]+1 - p1[1]) * 1. / (p2[0]+1 - p1[0])
   
def y_intercept(slope, p1) :
   return p1[1] - 1. * slope * p1[0]
   
def intersect(line1, line2) :
   min_allowed = 1e-5
   big_value = 1e10
   m1 = slope(line1[0], line1[1])
   b1 = y_intercept(m1, line1[0])
   m2 = slope(line2[0], line2[1])
   b2 = y_intercept(m2, line2[0])
   if abs(m1 - m2) < min_allowed :
      x = big_value
   else :
      x = (b2 - b1) / (m1 - m2)
   y = m1 * x + b1
   y2 = m2 * x + b2
   return (int(x),int(y))
   
def segment_intersect(line1, line2) :
   intersection_pt = intersect(line1, line2)
   if (line1[0][0] < line1[1][0]) :
      if intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0] :
         return None
   else :
      if intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0] :
         return None
   if (line2[0][0] < line2[1][0]) :
      if intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0] :
         return None
   else :
      if intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0] :
         return None
   return intersection_pt

def Lidar(x, y, angle):
    inputLines = [[pygame.math.Vector2(int(x+25), int(y+15)), pygame.math.Vector2(int(x+25), int(y+15))+pygame.math.Vector2(0, 700).rotate(-angle)],
                  [pygame.math.Vector2(int(x+25), int(y+15)), pygame.math.Vector2(int(x+25), int(y+15))+pygame.math.Vector2(0, -700).rotate(-angle)],
                  [pygame.math.Vector2(int(x+30), int(y+25)), pygame.math.Vector2(int(x+30), int(y+25))+pygame.math.Vector2(1200, 0).rotate(-angle)],
                  [pygame.math.Vector2(int(x+25), int(y+20)), pygame.math.Vector2(int(x+25), int(y))+pygame.math.Vector2(1200, 2000).rotate(-angle)],
                  [pygame.math.Vector2(int(x+25), int(y+20)), pygame.math.Vector2(int(x+25), int(y))+pygame.math.Vector2(1200, 980).rotate(-angle)],
                  [pygame.math.Vector2(int(x+25), int(y+20)), pygame.math.Vector2(int(x+25), int(y))+pygame.math.Vector2(1200, -2000).rotate(-angle)],
                  [pygame.math.Vector2(int(x+25), int(y+20)), pygame.math.Vector2(int(x+25), int(y))+pygame.math.Vector2(1200, -980).rotate(-angle)]]
    results = []
    for i in range(len(inputLines)):
        dist = math.inf
        inter = None
        index = None
        for j in range(len(lines)):
            intersetion = segment_intersect(lines[j], [[inputLines[i][0][0], inputLines[i][0][1]], [inputLines[i][1][0], inputLines[i][1][1]]])
            if(intersetion is not None):
                valDist = GetDistance(inputLines[i][0][0], inputLines[i][0][1], intersetion[0], intersetion[1])
                if(valDist <= dist):
                    dist = valDist
                    inter = intersetion
                    index = j
        if(inter != None):
            pygame.draw.lines(DISPLAYSURF, BACK, False, [(inputLines[i][0][0], inputLines[i][0][1]), (inter[0], inter[1])], 2)
            pygame.draw.circle(DISPLAYSURF, BLUE, inter, 6, 3)
            results.append(dist/1200)
    return results

def reLU(x):
    if(x < 0):
        return 0
    return x

def Sigmoid(x):
    return 1/(1 + np.exp(-x))

class Person:
    def __init__(self, x, y, tip, speed, idOfCar):
        self.x = x
        self.y = y
        self.angle = 0
        self.tip = tip
        self.score = 0
        self.speed = speed
        self.idOfCar = idOfCar
        self.liveState = True
        self.weightsHidden1 = [[round(random.uniform(-1, 1), 2) for _ in range(7)] for _ in range(6)]
        self.weightsHidden2 = [[round(random.uniform(-1, 1), 2) for _ in range(6)] for _ in range(6)]
        self.weightsOutput  = [[round(random.uniform(-1, 1), 2) for _ in range(6)] for _ in range(3)]
        self.biasesHidden1  = [round(random.uniform(-1, 1), 2)]*6
        self.biasesHidden2  = [round(random.uniform(-1, 1), 2)]*6
        self.biasesOutput   = [round(random.uniform(-1, 1), 2)]*6
        
        
    def DrawPerson(self):
        self.angle %= 360
        center = normalImg.get_rect().center
        rotated_image = pygame.transform.rotate(normalImg, self.angle)
        if(self.liveState == True):
            if(self.tip == 1):
                center = yourselfImg.get_rect().center
                rotated_image = pygame.transform.rotate(yourselfImg, self.angle)
            else:
                canChange = False
                maxi = -math.inf
                indexMaxi = -1
                for i in range(20):
                    if(maxi <= scores[i]):
                        maxi = scores[i]
                        indexMaxi = i
                if(self.idOfCar >= indexMaxi):
                    canChange = True
                if(canChange == True):
                    center = bestImg.get_rect().center
                    rotated_image = pygame.transform.rotate(bestImg, self.angle)
                else:
                    center = normalImg.get_rect().center
                    rotated_image = pygame.transform.rotate(normalImg, self.angle)
            new_rect = rotated_image.get_rect(center = center)
            pos = [new_rect.topleft[0]+self.x, new_rect.topleft[1]+self.y]
            DISPLAYSURF.blit(rotated_image, tuple(pos))
            for i in range(len(rects)):
                if(colision(self.x+25, self.y+25, 22, rects[i][0], rects[i][1], rects[i][2], rects[i][3]) == True):
                    self.liveState = False
                    break
            lidarOutput = Lidar(self.x, self.y, self.angle)
            self.predict()
        else:
            center = deadImg.get_rect().center
            rotated_image = pygame.transform.rotate(deadImg, self.angle)
            new_rect = rotated_image.get_rect(center = center)
            pos = [new_rect.topleft[0]+self.x, new_rect.topleft[1]+self.y]
            DISPLAYSURF.blit(rotated_image, tuple(pos))

    def ResetPerson(self):
        self.x = 180
        self.y = 640
        self.angle = 0
        self.score = 0
        self.lifeState = True
        scores[self.idOfCar] = 0

    def Left(self):
        if(self.liveState):
            self.angle += 4
            self.angle %= 360

    def Right(self):
        if(self.liveState):
            self.angle -= 4
            self.angle %= 360

    def Move(self):
        if(self.liveState):
            self.angle %= 360
            dx = math.cos(math.radians(self.angle))
            dy = math.sin(math.radians(self.angle))
            self.x = self.x+dx*self.speed
            self.y = self.y-dy*self.speed
            self.score += 1
            scores[self.idOfCar] = self.score

    def predict(self):
        self.angle %= 360
        Encoded = Lidar(self.x, self.y, self.angle)
        NeuralNetInput = Encoded
        HiddenLayer1 = [float(0)]*6
        HiddenLayer2 = [float(0)]*6
        Output       = [float(0)]*3
        # Hidden Layer 1
        for index in range(6):
            for i in range(len(Encoded)):
                HiddenLayer1[index] = HiddenLayer1[index]+(Encoded[i]*self.weightsHidden1[index][i])
            HiddenLayer1[index] = reLU(HiddenLayer1[index]+self.biasesHidden1[index])

        # Hidden Layer 2
        for index in range(6):
            for i in range(len(HiddenLayer1)):
                HiddenLayer2[index] = HiddenLayer2[index]+(HiddenLayer1[i]*self.weightsHidden2[index][i])
            HiddenLayer2[index] = reLU(HiddenLayer2[index]+self.biasesHidden2[index])

        # Output Layer
        for index in range(3):
            for i in range(len(HiddenLayer2)):
                Output[index] = Output[index]+(HiddenLayer2[i]*self.weightsOutput[index][i])
            Output[index] = reLU(Output[index]+self.biasesOutput[index])

        index = -1
        maxi = -math.inf
        for i in range(len(Output)):
            if(Sigmoid(Output[i]) > maxi):
                maxi = Sigmoid(Output[i])
                index = i
        lookup = {
            0:'left',
            1:'right',
            2:'nothing'
        }
        if(lookup[index] == 'left'):
            self.Left()
        elif(lookup[index] == 'right'):
            self.Right()
    def copyChildren(self,template1WeightsHidden1,template1WeightsHidden2,template1WeightsOutput,
                          template1BiasesHidden1 ,template1BiasesHidden2, template1BiasesOutput,
                          template2WeightsHidden1,template2WeightsHidden2,template2WeightsOutput,
                          template2BiasesHidden1 ,template2BiasesHidden2, template2BiasesOutput,
                          template3WeightsHidden1,template3WeightsHidden2,template3WeightsOutput,
                          template3BiasesHidden1 ,template3BiasesHidden2, template3BiasesOutput):
        # Change Weights
        for index in range(6):
            for i in range(7):
                number = random.uniform(0, 1)
                if(number >= 0.00 and number <= 0.33): self.weightsHidden1[index][i] = template1WeightsHidden1[index][i];
                if(number >= 0.34 and number <= 0.66): self.weightsHidden1[index][i] = template2WeightsHidden1[index][i];
                if(number >= 0.67 and number <= 1.00): self.weightsHidden1[index][i] = template3WeightsHidden1[index][i];
                coin = random.uniform(-1, 1)
                if(coin >= 0.0):
                    self.weightsHidden1[index][i] += random.uniform(-0.5, 0.5)
        for index in range(6):
            for i in range(6):
                number = random.uniform(0, 1)
                if(number >= 0.00 and number <= 0.33): self.weightsHidden2[index][i] = template1WeightsHidden2[index][i];
                if(number >= 0.34 and number <= 0.66): self.weightsHidden2[index][i] = template2WeightsHidden2[index][i];
                if(number >= 0.67 and number <= 1.00): self.weightsHidden2[index][i] = template3WeightsHidden2[index][i];
                coin = random.uniform(-1, 1)
                if(coin >= 0.0):
                    self.weightsHidden2[index][i] += random.uniform(-0.5, 0.5)
        for index in range(3):
            for i in range(6):
                number = random.uniform(0, 1)
                if(number >= 0.00 and number <= 0.33): self.weightsOutput[index][i] = template1WeightsOutput[index][i];
                if(number >= 0.34 and number <= 0.66): self.weightsOutput[index][i] = template2WeightsOutput[index][i];
                if(number >= 0.67 and number <= 1.00): self.weightsOutput[index][i] = template3WeightsOutput[index][i];
                coin = random.uniform(-1, 1)
                if(coin >= 0.0):
                    self.weightsOutput[index][i] += random.uniform(-0.5, 0.5)

        # Change Biases
        for i in range(6):
            number = random.uniform(0, 1)
            if(number >= 0.00 and number <= 0.33): self.biasesHidden1[i] = template1BiasesHidden1[i];
            if(number >= 0.34 and number <= 0.66): self.biasesHidden1[i] = template2BiasesHidden1[i];
            if(number >= 0.67 and number <= 1.00): self.biasesHidden1[i] = template3BiasesHidden1[i];
            coin = random.uniform(-1, 1)
            if(coin >= 0.0):
                self.biasesHidden1[i] += random.uniform(-0.5, 0.5)

        for i in range(6):
            number = random.uniform(0, 1)
            if(number >= 0.00 and number <= 0.33): self.biasesHidden2[i] = template1BiasesHidden2[i];
            if(number >= 0.34 and number <= 0.66): self.biasesHidden2[i] = template2BiasesHidden2[i];
            if(number >= 0.67 and number <= 1.00): self.biasesHidden2[i] = template3BiasesHidden2[i];
            coin = random.uniform(-1, 1)
            if(coin >= 0.0):
                self.biasesHidden2[i] += random.uniform(-0.5, 0.5)

        for i in range(3):
            number = random.uniform(0, 1)
            if(number >= 0.00 and number <= 0.33): self.biasesOutput[i] = template1BiasesOutput[i];
            if(number >= 0.34 and number <= 0.66): self.biasesOutput[i] = template2BiasesOutput[i];
            if(number >= 0.67 and number <= 1.00): self.biasesOutput[i] = template3BiasesOutput[i];
            coin = random.uniform(-1, 1)
            if(coin >= 0.0):
                self.biasesOutput[i] += random.uniform(-0.5, 0.5)
            
geneticPersons = []
for i in range(15):
    geneticPersons.append(Person(180, 640, 2, 6, i))

def NewGeneration():
    global gen
    gen += 1
    bestScore = -math.inf
    indexBest1 = -1
    for i in range(15):
        if(bestScore <= geneticPersons[i].score):
            bestScore = geneticPersons[i].score
            indexBest1 = i
    bestScore = -math.inf
    indexBest2 = -1
    for i in range(15):
        if(bestScore <= geneticPersons[i].score and i != indexBest1):
            bestScore = geneticPersons[i].score
            indexBest2 = i
    bestScore = -math.inf
    indexBest3 = -1
    for i in range(15):
        if(bestScore <= geneticPersons[i].score and i != indexBest1 and i != indexBest2):
            bestScore = geneticPersons[i].score
            indexBest3 = i
            
    for i in range(15):
        geneticPersons[i].ResetPerson()
    for i in range(15):
        if(i != indexBest1 and i != indexBest2 and i != indexBest3):
            template1 = geneticPersons[indexBest1]
            template2 = geneticPersons[indexBest2]
            template3 = geneticPersons[indexBest3]
            geneticPersons[i].copyChildren(template1.weightsHidden1,template1.weightsHidden2,template1.weightsOutput,
                                            template1.biasesHidden1 ,template1.biasesHidden2, template1.biasesOutput,
                                            template2.weightsHidden1,template2.weightsHidden2,template2.weightsOutput,
                                            template2.biasesHidden1 ,template2.biasesHidden2, template2.biasesOutput,
                                            template3.weightsHidden1,template3.weightsHidden2,template3.weightsOutput,
                                            template3.biasesHidden1 ,template3.biasesHidden2, template3.biasesOutput)
        geneticPersons[i].liveState = True

def NumberOfPersonsAlive():
    rez = 0
    for i in range(15):
        if(geneticPersons[i].liveState):
            rez += 1
    return rez

def DrawScene():
    DrawTrack()
    for i in range(15):
        geneticPersons[i].DrawPerson()

    font = pygame.font.Font("freesansbold.ttf", 53)
    text = font.render('AI Learns To Walk', True, YELLOW)
    textRect = text.get_rect()
    textRect.center = (1200//2, 340)
    DISPLAYSURF.blit(text, textRect) 

    font = pygame.font.Font("freesansbold.ttf", 24)
    text = font.render('Generation: ' + str(gen), True, BLACK)
    textRect = text.get_rect()
    textRect.center = (1200//2, 385)
    DISPLAYSURF.blit(text, textRect)

    font = pygame.font.Font("freesansbold.ttf", 24)
    text = font.render('Remaning: ' + str(NumberOfPersonsAlive()), True, BLACK)
    textRect = text.get_rect()
    textRect.center = (1200//2, 430)
    DISPLAYSURF.blit(text, textRect)

    font = pygame.font.Font("freesansbold.ttf", 24)
    text = font.render('Press 1 To Save', True, BLACK)
    textRect = text.get_rect()
    textRect.center = (125, 30)
    DISPLAYSURF.blit(text, textRect)

    text = font.render('Press 2 To Load', True, BLACK)
    textRect = text.get_rect()
    textRect.center = (125, 60)
    DISPLAYSURF.blit(text, textRect)

def DrawTrack():
    pygame.draw.rect(DISPLAYSURF,GREY,(25,590,1155,150))
    pygame.draw.rect(DISPLAYSURF,GREY,(25,10,1155,150))
    pygame.draw.rect(DISPLAYSURF,GREY,(1030,160,150,430))
    pygame.draw.rect(DISPLAYSURF,GREY,(25,160,150,550))
    pygame.draw.rect(DISPLAYSURF,GREY,(25,540,150,150))

    pygame.draw.rect(DISPLAYSURF,WHITE,(325,590,80,150))

    finishIndex = 0
    y_offset = 595
    for i in range(7):
        x_offset = 325
        for j in range(4):
            if(finishIndex%2 == 0):
                pygame.draw.rect(DISPLAYSURF,BLACK,(x_offset,y_offset,20,20))
            else:
                pygame.draw.rect(DISPLAYSURF,WHITE,(x_offset,y_offset,20,20))
            finishIndex += 1
            x_offset += 20
        y_offset += 20
        finishIndex += 1
    pass

def SaveGeneticModel():
    bestScore = -math.inf
    indexBest = -1
    for i in range(15):
        if(bestScore <= geneticPersons[i].score):
            bestScore = geneticPersons[i].score
            indexBest = i
    bestPerson = geneticPersons[indexBest]
    biasesHidden1File = open("GeneticModel/biasesHidden1.txt", "w+")
    biasesHidden2File = open("GeneticModel/biasesHidden2.txt", "w+")
    biasesOutputFile = open("GeneticModel/biasesOutput.txt", "w+")
    weightsHidden1File = open("GeneticModel/weightsHidden1.txt", "w+")
    weightsHidden2File = open("GeneticModel/weightsHidden2.txt", "w+")
    weightsOutputFile = open("GeneticModel/weightsOutput.txt", "w+")
    for i in range(len(bestPerson.biasesHidden1)): biasesHidden1File.write(str(bestPerson.biasesHidden1[i]) + "\n");
    for i in range(len(bestPerson.biasesHidden2)): biasesHidden2File.write(str(bestPerson.biasesHidden2[i]) + "\n");
    for i in range(len(bestPerson.biasesOutput)): biasesOutputFile.write(str(bestPerson.biasesOutput[i]) + "\n");
    for index in range(6):
        for i in range(7): weightsHidden1File.write(str(bestPerson.weightsHidden1[index][i]) + "\n");
    for index in range(6):
        for i in range(6): weightsHidden2File.write(str(bestPerson.weightsHidden2[index][i]) + "\n");
    for index in range(3):
        for i in range(6): weightsOutputFile.write(str(bestPerson.weightsOutput[index][i]) + "\n");
    biasesHidden1File.close()
    biasesHidden2File.close()
    biasesOutputFile.close()
    weightsHidden1File.close()
    weightsHidden2File.close()
    weightsOutputFile.close()

def GetValuesFromFile(fileName):
    f = open(fileName, "r+")
    results = []
    row = ""
    while True:
        c = f.read(1)
        if not c:
            break
        if(c != '\n'):
            row += c
        if(c == '\n'):
            results.append(float(row))
            row = ""
    f.close()
    return results

def LoadGeneticModel():
    for i in range(15):
        geneticPersons[i].ResetPerson()
    for i in range(15):
        geneticPersons[i].liveState = True
    LoadelPerson = geneticPersons[0]
    LoadelPerson.biasesHidden1 = GetValuesFromFile("GeneticModel/biasesHidden1.txt")
    LoadelPerson.biasesHidden2 = GetValuesFromFile("GeneticModel/biasesHidden2.txt")
    LoadelPerson.biasesOutput = GetValuesFromFile("GeneticModel/biasesOutput.txt")
    globalIndex = 0
    for index in range(6):
        for i in range(7):
             LoadelPerson.weightsHidden1[index][i] = GetValuesFromFile("GeneticModel/weightsHidden1.txt")[globalIndex]
             globalIndex += 1
    globalIndex = 0
    for index in range(6):
        for i in range(6):
            LoadelPerson.weightsHidden2[index][i] = GetValuesFromFile("GeneticModel/weightsHidden1.txt")[globalIndex]
            globalIndex += 1
    globalIndex = 0
    for index in range(3):
        for i in range(6):
            LoadelPerson.weightsOutput[index][i] = GetValuesFromFile("GeneticModel/weightsOutput.txt")[globalIndex]
            globalIndex += 1
    for i in range(1):
        geneticPersons[i].biasesHidden1 = LoadelPerson.biasesHidden1
        geneticPersons[i].biasesHidden2 = LoadelPerson.biasesHidden2
        geneticPersons[i].biasesOutput = LoadelPerson.biasesOutput

        geneticPersons[i].weightsHidden1 = LoadelPerson.weightsHidden1
        geneticPersons[i].weightsHidden2 = LoadelPerson.weightsHidden2
        geneticPersons[i].weightsOutput = LoadelPerson.weightsOutput
    print(LoadelPerson)

while True:
    clk += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif(event.type == pygame.KEYUP):
            if(event.key == pygame.K_1):
                SaveGeneticModel()
            elif(event.key == pygame.K_2):
                LoadGeneticModel()

    if(NumberOfPersonsAlive() != 0):
        for i in range(15):
            geneticPersons[i].Move()
    else:
        NewGeneration()
    DISPLAYSURF.fill(BUTTON)
    DrawScene()
    pygame.display.update()
    mouse_x, mouse_y = pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1]

pygame.quit()
quit()

# TO DO:
#   - add collision - done
#   - add lidar - done
#   - encode lidar output - done
#   - add individual nn - done
#   - add genetic algo - done
#   - add better genetic algo - done
#   - fix lidar - done
#   - add loading/saving genetic model - working

# --------- NEURAL NET PARAMETERS ---------
# iL  = 9 (the lidar output & the angle)
# hL1 = 6
# hL2 = 6
# oL  = 3 (left, right, no action)
