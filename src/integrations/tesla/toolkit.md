---
name: tesla
summary: "the cars: defrost, climate, trunk, navigation"
commands: [tesla_max_defrost, tesla_air_con_on, tesla_air_con_off, tesla_open_trunk, tesla_navigate]
---
You can interact with the families car using these commands (use this syntax exactly):

`tesla_max_defrost(vehicle_id: str)`
`tesla_air_con_on(vehicle_id: str)`
`tesla_air_con_off(vehicle_id: str)`
`tesla_open_trunk(vehicle_id: str)`
`tesla_navigate(vehicle_id: str, destination: str)` - sends a destination to the car's navigation; `destination` is a street address, a place name with its town, or "latitude,longitude"

The cars and their ids are in the CURRENT TESLA CAR INFO section of your prompt.
