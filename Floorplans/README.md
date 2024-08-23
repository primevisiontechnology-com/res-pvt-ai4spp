# floorplan-factory-USA

Repository with all the floorplan developmens from the IST team.

## Guidelines

A floorplan consist of the following components:
* Zones
* Nodes
    * init
    * target
    * entry_and_exit
* Connections
    * entry
    * exit
    * control points
* Special Nodes:
    * Induction Node
    * Homing Node

All robots work without config changes to the following gates:
* input_gate_1
* cross_gate
* output_gate_1

Bin Sizes:
* Gaylord long side: 1.22
* Gaylord short side: 1.02
* Hampers short side: 0.8
* Hampers long side: 1.22

Row widths:
* Gaylord row + robot: 2.76
* Driving row : 1.2
* Finger row: 1.6
* Hamper row + robot: 2.36

Turn Radius:
* 

Clear Distances:
* Corner nodes must have a clear distance of 0.3
* Clear distance + distance to next node should be 1.3

Bin numbers:
* Left <=> Right
* Up <=> Down
* Start from 001
* Reject skips 1 bin 33 => reject => 35

Known issues:
* Entry-and-exit nodes can not be corner nodes


Throughput gains:
* equal distance between robots
* allow for automatic handover (robot does not stop on infeed)
* shortcuts

## Repository rules

* Folder structure:
```bash
├── dual-infeed # Delft accepted floorplan types
├── parts # IST specific parts used for quick development
│   ├── crossings
│   ├── fingers
│   └── infeeds
└── USPS # customer and/or project specific developments
    ├── 0022_Memphis # top level system number and location
    │   ├── floorplan.json # active
    │   ├── floorplan_develop.json # current developments
    │   ├── sortplan.json # "active"
    │   ├── fp-factory # floorplans used for FAT's
    │   ├── stable-releases
    │   │   ├── floorplan-v1_0.json # -v{top}_{sub}.json
    │   │   ├── floorplan-v2_0.json
    │   │   └── sortplan.json
    │   └── unstable
    │       ├── floorplan-v0_unstable_1.json # -v{top}_unstable_{sub}.json
    │       ├── floorplan-v0_unstable_2.json
    │       ├── floorplan-v2_unstable_1.json
    │       └── sortplan.json
```

* Floorplans should not be pushed shifted to this repository


## Infeedcell Loading
Use the following paramaters at the infeed cell to tweak the load betweeen infeed cells.
```
"target_reservation_cost_linear": 32,
"target_reservation_cost_quad": 0,
"virtual_target_reservation_cost_linear": 0,"virtual_target_reservation_cost_quad": 0,
"check_in_reservation_cost_linear": 4,
"check_in_reservation_cost_quad": 0,
```
## Authors and acknowledgment
Timo Thans
```
t.thans@primevision.com
+484 828 2357
```
Brandon Kelly
```
b.kelly@primevision.com
```