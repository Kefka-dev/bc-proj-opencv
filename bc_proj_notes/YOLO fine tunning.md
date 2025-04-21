**train 5**
NEDOBRE LEBO CH15 MA SPATNE ANOTACIE (**neskus zmazat, porovnaj to**)
- 3500 obrazkov cca
- zaklad yolo11l pretrained na coco datasete

**train 4** 
NEDOBRE LEBO CH15 MA SPATNE ANOTACIE
- 3500 obrazkov cca
- zaklad je yolo11m pretrained na coco datasete
- 

**train 6**  
 - from scratch training len s mojimy datami
 - baseline bude yolo11m
	 - cca 3minuty na epochu
 - merged_dataset2 
	 - vsetky channels okrem ch3(nekonzistentné bounding boxy, konkretne niesu anotovane sposobom amodal(tj bounding box je aj tam kde je okulzia ))
**train 7** ==TODO==
- from scratch training len s mojimy datami
 - baseline bude yolo11l
	 - prvy test bol zatial tak 10 min na epochu (skoro 3x tak casovo narocne ako t6)
 - merged_dataset2 
	 - idealne by to chcelo opravit ch3 este, lebo trenovat toto 2x je waste casu a elektriky
 - 

`data.yaml` štruktúra datasetu2
```yaml
#path to the dataset root, our case is merged_dataset
#if we place the data.yaml inside of it the path will be "."
#personaly had some cash issues with ".", so i used absolute path to the dataset root
path: .

#paths relative to the root of the dataset defined in path
train: train/images
val: val/images
test: test/images

#number of classes
nc: 1

names: ['person']
```

===UPDATE 21.4.2025====
V **dataset2** Všetky channels su anotated, z kažedeho zhruba 500 frames.
- **CH3** nie je v tomto datasete zahrnutý, kedze bol anotovaný ako prvý, čiže nesedí spôsob anotácie s ostatnými, t.j. nie je anotated sposobom amodal bounding boxes (bounding box sa kreslí aj cez occulsion, nie len viditelné časti)
	- prečo? v tej dobe som ešte nemal jasno ako to vlasne chcem robiť, tj musím to preanotovať, hrubá časť práce je tam ale spravená
- 