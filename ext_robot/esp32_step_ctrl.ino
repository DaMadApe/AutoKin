#include <AccelStepper.h>
#include <MultiStepper.h>

#define PASO_M1 4   //PASO a pin digital 4
#define DIR_M1 2    //DIR a pin digital 2
#define PASO_M2 23  //PASO a pin digital 23
#define DIR_M2 22   //DIR a pin digital 22
#define PASO_M3 26  //PASO a pin digital 26
#define DIR_M3 25   //DIR a pin digital 25
#define PASO_M4 33  //PASO a pin digital 33
#define DIR_M4 32  //DIR a pin digital 32
#define PASO_F 16
#define DIR_F 17

#define MIN_PASOS 50 //Valor real: MIN_PASOS*2
#define BUFF_SIZE 4*4

const int width_param = 30;
const int vel_param = 100;

String cadena = ""; // Señal de control proveniente de la CPU
int pos[3] = {0,0,0};
long npasos[5]; // = {0,0,0,0};
byte in_buffer[BUFF_SIZE] = {0};
bool inicio = false;

AccelStepper stepper1(AccelStepper::DRIVER, PASO_M1, DIR_M1);
AccelStepper stepper2(AccelStepper::DRIVER, PASO_M2, DIR_M2);
AccelStepper stepper3(AccelStepper::DRIVER, PASO_M3, DIR_M3);
AccelStepper stepper4(AccelStepper::DRIVER, PASO_M4, DIR_M4);
AccelStepper stepperF(AccelStepper::DRIVER, PASO_F, DIR_F);

MultiStepper steppers;

void setup()
{  
    Serial.begin(115200);

    // Ancho de pulso
    stepper1.setMinPulseWidth(width_param);
    stepper2.setMinPulseWidth(width_param);
    stepper3.setMinPulseWidth(width_param);
    stepper4.setMinPulseWidth(width_param);
    stepperF.setMinPulseWidth(width_param);

    // Velocidad Máxima
    stepper1.setMaxSpeed(vel_param);
    stepper2.setMaxSpeed(vel_param);
    stepper3.setMaxSpeed(vel_param);
    stepper4.setMaxSpeed(vel_param);
    stepperF.setMaxSpeed(vel_param);
 
    // Motores
    steppers.addStepper(stepper1);
    steppers.addStepper(stepper2);
    steppers.addStepper(stepper3);
    steppers.addStepper(stepper4);
    steppers.addStepper(stepperF);

    npasos[4] = MIN_PASOS;
}

void loop() {
  if(Serial.available()){

    Serial.readBytes(in_buffer, BUFF_SIZE);

    int* intpasos = (int*) in_buffer;
    for(int i=0; i<4; i++){
      npasos[i] = (long) intpasos[i];
    }
    npasos[4] = -1*npasos[4];

    steppers.moveTo(npasos);
    steppers.runSpeedToPosition();

    Serial.write('x');
    Serial.flush();
  }
}