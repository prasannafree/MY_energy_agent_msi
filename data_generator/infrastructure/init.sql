-- ==========================================================
-- BUILDING MANAGEMENT DATABASE
-- Agentic AI for Smart Buildings
-- Version : 2.0
-- ==========================================================


-- ==========================================================
-- DEVICE TYPE ENUM
-- Restricts device_type to valid values
-- ==========================================================

CREATE TYPE device_type_enum AS ENUM (

    'temperature',
    'humidity',
    'co2',
    'occupancy',
    'power',
    'voltage',
    'current',
    'energy',
    'hvac'

);



-- ==========================================================
-- DEVICE STATUS ENUM
-- ==========================================================

CREATE TYPE device_status_enum AS ENUM (

    'ACTIVE',
    'OFFLINE',
    'FAULTY',
    'MAINTENANCE'

);



-- ==========================================================
-- BUILDINGS
-- ==========================================================

CREATE TABLE buildings (

    building_id SERIAL PRIMARY KEY,

    building_name VARCHAR(100) NOT NULL,

    location VARCHAR(200),

    timezone VARCHAR(50),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);



-- ==========================================================
-- ZONES
-- ==========================================================

CREATE TABLE zones (

    zone_id SERIAL PRIMARY KEY,

    building_id INTEGER NOT NULL,

    zone_name VARCHAR(100) NOT NULL,

    floor_number INTEGER,

    area_sq_m NUMERIC(8,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (building_id)
        REFERENCES buildings(building_id)
        ON DELETE CASCADE

);



-- ==========================================================
-- DEVICES
-- ==========================================================

CREATE TABLE devices (

    device_id SERIAL PRIMARY KEY,

    zone_id INTEGER NOT NULL,

    device_name VARCHAR(100) NOT NULL,

    device_type device_type_enum NOT NULL,

    manufacturer VARCHAR(100),

    status device_status_enum DEFAULT 'ACTIVE',

    installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (zone_id)
        REFERENCES zones(zone_id)
        ON DELETE CASCADE

);



-- ==========================================================
-- TELEMETRY
-- One row = One measurement from one device
-- ==========================================================

CREATE TABLE telemetry (

    telemetry_id BIGSERIAL PRIMARY KEY,

    device_id INTEGER NOT NULL,

    sensor_value DOUBLE PRECISION NOT NULL,

    unit VARCHAR(20) NOT NULL,

    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (device_id)
        REFERENCES devices(device_id)
        ON DELETE CASCADE

);



-- ==========================================================
-- INDEXES
-- ==========================================================

CREATE INDEX idx_zone_building
ON zones(building_id);

CREATE INDEX idx_device_zone
ON devices(zone_id);

CREATE INDEX idx_device_type
ON devices(device_type);

CREATE INDEX idx_telemetry_device
ON telemetry(device_id);

CREATE INDEX idx_telemetry_time
ON telemetry(recorded_at);



-- ==========================================================
-- SEED DATA
-- ==========================================================

--------------------------------------------------------------
-- BUILDING
--------------------------------------------------------------

INSERT INTO buildings
(
    building_name,
    location,
    timezone
)
VALUES
(
    'Engineering Building',
    'Chennai',
    'Asia/Kolkata'
);



--------------------------------------------------------------
-- ZONES
--------------------------------------------------------------

INSERT INTO zones
(
    building_id,
    zone_name,
    floor_number,
    area_sq_m
)
VALUES

(1,'Zone 1',1,120.50),
(1,'Zone 2',1,95.00),
(1,'Zone 3',2,150.75);



--------------------------------------------------------------
-- DEVICES : ZONE 1
--------------------------------------------------------------

INSERT INTO devices
(
    zone_id,
    device_name,
    device_type,
    manufacturer
)
VALUES

(1,'Temperature Sensor Z1','temperature','Bosch'),
(1,'Humidity Sensor Z1','humidity','Bosch'),
(1,'CO2 Sensor Z1','co2','Honeywell'),
(1,'Occupancy Sensor Z1','occupancy','Siemens'),
(1,'Power Meter Z1','power','Schneider');



--------------------------------------------------------------
-- DEVICES : ZONE 2
--------------------------------------------------------------

INSERT INTO devices
(
    zone_id,
    device_name,
    device_type,
    manufacturer
)
VALUES

(2,'Temperature Sensor Z2','temperature','Bosch'),
(2,'Humidity Sensor Z2','humidity','Bosch'),
(2,'CO2 Sensor Z2','co2','Honeywell'),
(2,'Occupancy Sensor Z2','occupancy','Siemens'),
(2,'Power Meter Z2','power','Schneider');



--------------------------------------------------------------
-- DEVICES : ZONE 3
--------------------------------------------------------------

INSERT INTO devices
(
    zone_id,
    device_name,
    device_type,
    manufacturer
)
VALUES

(3,'Temperature Sensor Z3','temperature','Bosch'),
(3,'Humidity Sensor Z3','humidity','Bosch'),
(3,'CO2 Sensor Z3','co2','Honeywell'),
(3,'Occupancy Sensor Z3','occupancy','Siemens'),
(3,'Power Meter Z3','power','Schneider');