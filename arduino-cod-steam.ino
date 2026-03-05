

#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include <Stepper.h>
const int stepperOrigin = 0;
const int steps = 256;  // change this to fit the number of steps per revolution

// ULN2003 Motor Driver Pins
#define IN1 11
#define IN2 10
#define IN3 9
#define IN4 8
#define IRsensor 3
// initialize the stepper library
Stepper myStepper(steps, IN1, IN3, IN2, IN4);

#define DHTPIN 2     // Digital pin connected to the DHT sensor 

#define DHTTYPE    DHT22     // DHT 22 (AM2302)

DHT_Unified dht(DHTPIN, DHTTYPE);

uint32_t delayMS;

bool movedForward = false;

void setup() {
  Serial.begin(9600);
  pinMode(IRsensor, INPUT);
  dht.begin();
  Serial.println(F("DHTxx Unified Sensor Example"));
  myStepper.setSpeed(20);
  // Print temperature sensor details.
  sensor_t sensor;
  dht.temperature().getSensor(&sensor);
  Serial.println(F("------------------------------------"));
  Serial.println(F("Temperature Sensor"));
  Serial.print  (F("Sensor Type: ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:  ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:   ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:   ")); Serial.print(sensor.max_value); Serial.println(F("°C"));
  Serial.print  (F("Min Value:   ")); Serial.print(sensor.min_value); Serial.println(F("°C"));
  Serial.print  (F("Resolution:  ")); Serial.print(sensor.resolution); Serial.println(F("°C"));
  Serial.println(F("------------------------------------"));
  // Print humidity sensor details.
  dht.humidity().getSensor(&sensor);
  Serial.println(F("Humidity Sensor"));
  Serial.print  (F("Sensor Type: ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:  ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:   ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:   ")); Serial.print(sensor.max_value); Serial.println(F("%"));
  Serial.print  (F("Min Value:   ")); Serial.print(sensor.min_value); Serial.println(F("%"));
  Serial.print  (F("Resolution:  ")); Serial.print(sensor.resolution); Serial.println(F("%"));
  Serial.println(F("------------------------------------"));
  // Set delay between sensor readings based on sensor details.
  delayMS = sensor.min_delay / 1000;
}

void loop() {
  delay(delayMS);
  int flame = digitalRead(IRsensor);
  sensors_event_t tempEvent;
  sensors_event_t humEvent;

  // Read temperature
  dht.temperature().getEvent(&tempEvent);
  if (isnan(tempEvent.temperature)) {
    Serial.println(F("Error reading temperature!"));
  } else {
    Serial.print(F("Temperature: "));
    Serial.print(tempEvent.temperature);
    Serial.print(F("°C     "));
  }

  // Read humidity
  dht.humidity().getEvent(&humEvent);
  if (isnan(humEvent.relative_humidity)) {
    Serial.println(F("Error reading humidity!"));
  } else {
    Serial.print(F("Humidity: "));
    Serial.print(humEvent.relative_humidity);
    Serial.print(F("%     "));
  }
  Serial.print("IR sensor reading: ");
  Serial.print(flame);
  Serial.print("     ");
  if (flame == 1 && movedForward == false && tempEvent.temperature > 30.0) {
    Serial.println("Conclusion: fire");
     myStepper.step(steps);
     movedForward = true;
  }
  else{
    Serial.println("Conclusion: no fire");
  }
  if (flame == 0 && movedForward == true && tempEvent.temperature <= 30.0){
    
    myStepper.step(-steps);
    movedForward = false;
  }
  Serial.println("__________________________________________________________________");
  }

