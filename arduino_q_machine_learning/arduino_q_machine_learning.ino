//##################################################################
//###                  Arduino Q-Learning                        ###
//###  The play-field is 4x4 fields, the software will search    ###
//###              the best way to the target                    ###
//##################################################################

//#include <Adafruit_GFX.h>    // Core graphics library
#include <Adafruit_TFTLCD.h> // Hardware-specific library
// The control pins for the LCD can be assigned to any digital or
// analog pins...but we'll use the analog pins as this allows us to
// double up the pins with the touch screen (see the TFT paint example).
#define LCD_CS A3 // Chip Select goes to Analog 3
#define LCD_CD A2 // Command/Data goes to Analog 2
#define LCD_WR A1 // LCD Write goes to Analog 1
#define LCD_RD A0 // LCD Read goes to Analog 0
#define LCD_RESET A4 // Can alternately just connect to Arduino's reset pin
// When using the BREAKOUT BOARD only, use these 8 data lines to the LCD:
// For the Arduino Uno, Duemilanove, Diecimila, etc.:
//   D0 connects to digital pin 8  (Notice these are
//   D1 connects to digital pin 9   NOT in order!)
//   D2 connects to digital pin 2
//   D3 connects to digital pin 3
//   D4 connects to digital pin 4
//   D5 connects to digital pin 5
//   D6 connects to digital pin 6
//   D7 connects to digital pin 7
// For the Arduino Mega, use digital pins 22 through 29
// (on the 2-row header at the end of the board).
// Assign human-readable names to some common 16-bit color values:
#define BLACK       0x0000
#define BLUE        0x001F
#define RED         0xF800
#define GREEN       0x07E0
#define CYAN        0x07FF
#define MAGENTA     0xF81F
#define YELLOW      0xFFE0
#define WHITE       0xFFFF
#define ORANGE      0xFBE0
#define GRAY        0x7BEF
#define NAVY        0x000F
#define DARKGREEN   0x03E0
#define DARKCYAN    0x03EF
#define MAROON      0x7800
#define PURPLE      0x780F
#define OLIVE       0x7BE0
#define LIGHTGRAY   0xC618
#define DARKGRAY    0x7BEF
#define GREENYELLOW 0xAFE5

Adafruit_TFTLCD tft(LCD_CS, LCD_CD, LCD_WR, LCD_RD, LCD_RESET);
// If using the shield, all control and data lines are fixed, and
// a simpler declaration can optionally be used:
// Adafruit_TFTLCD tft;

//Display Size:
int x_size = 240;
int y_size = 320;
int rotation = 0;
int size_fields = 60;//tft pixel
int delay_ms = 100;
//-------------------------------------------------------------------------------
#define total_fields 16
#define lines_of_fields 4
#define culumns_of_fields 4

int rewards[total_fields][total_fields] = {};//create reward field by algo
int weights [total_fields][total_fields] = {};
//-----------------------------------------
int start_field = 0; // the start position
int target_field = 14; // target position
int current_field = start_field;
int last_field = 0; // only for display update
int action;
int random_number;
int maximum_weight;
int n = 0;
int next_field;

//Version:
String sw_version = "Version: 0.1-Beta";
//#############################################################################################################
void setup() {

  tft.reset();
  tft.begin(0x9341);
  tft.setRotation(rotation);
  tft.fillScreen(BLACK);
  ScreenText(WHITE, 5, 5, 2 , sw_version);
  delay(2000);
  tft.fillScreen(BLACK);
  Serial.begin(9600);

  // init fields:
  create_reward_fields();
  //show_rewards();
  create_target_field(target_field);
  //show_rewards();
  create_nogo_field(10);
  create_nogo_field(13);
  show_rewards();
  init_weights_by_zero();
  set_field_frames();

  delay(1000);
}
//#############################################################################################################
void loop() {

  Serial.println(F("Start:"));

  while (n < 6) { // try's to find the best way

    while (current_field != target_field) { // search for target field by random action

      set_field_information(current_field, 1);

      while (rewards[current_field][random_number] < 0) { // search for neighbor by random
        random_number = random(0, (total_fields)); //  random(min, max)
      }

      action = random_number;
      maximum_weight = weights[action][0];

      for (int i = 1; i < total_fields; i++) {
        if (weights[action][i] > maximum_weight) {
          maximum_weight = weights[action][i];
        }
      }

      weights[current_field][action] = rewards[current_field][action] + 0.8 * maximum_weight;
      current_field = action;
      delay(delay_ms);
    }

    set_field_information(target_field, 1);
    delay(delay_ms);
    n++;
    current_field = 0;

  }

  //--------------------------------------------

  current_field = 0;
  while (current_field != target_field) { // search for best way by weights table

    set_field_information(current_field, 1);
    maximum_weight = weights[current_field][0];
    next_field = 0;

    for (int i = 1; i < total_fields; i++) {
      if (weights[current_field][i] > maximum_weight) {
        maximum_weight = weights[current_field][i];
        next_field = i;
      }
    }

    current_field = next_field;
    delay(delay_ms);
  }

  //--------------------------------------------
  set_field_information(target_field, 1);

  Serial.println(F("End"));
  show_weights();

  delay(delay_ms);
  set_way_information();//best way

  for (;;) {
    // endless loop
  }
}
//#############################################################################################################
void create_reward_fields() {

  for ( int y = 0; y < total_fields; y++) {
    for ( int x = 0; x < total_fields; x++) {
      rewards[y][x] = -1; //default value

      if (x == y + culumns_of_fields || x == y - culumns_of_fields) rewards[y][x] = 0; // 4 columns neighbor

      if (x == y + 1) { // direct neighbor ?
        int neighbor_1 = x % culumns_of_fields; //modulo
        int neighbor_2 = y % culumns_of_fields;
        if (abs(neighbor_1 - neighbor_2) == 1)  rewards[y][x] = 0; // both values in the same column
      }

      if (x == y - 1) { // direct neighbor ?
        int neighbor_1 = x % culumns_of_fields; //modulo
        int neighbor_2 = y % culumns_of_fields;
        if (abs(neighbor_1 - neighbor_2) == 1)  rewards[y][x] = 0; // both values in the same column
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------------------
void create_target_field(int target) {

  for ( int y = 0; y < total_fields; y++) {
    if (rewards[y][target] == 0) rewards[y][target] = 100;
  }
}
//--------------------------------------------------------------------------------------------------------------
void create_nogo_field(int nogo) {

  for ( int y = 0; y < total_fields; y++) {
    rewards[y][nogo] = -1;
  }
}
//--------------------------------------------------------------------------------------------------------------
void init_weights_by_zero() {

  Serial.println(F("Init weights by zero"));
  for ( int y = 0; y < total_fields; y++) {
    for ( int x = 0; x < total_fields; x++) {
      weights[y][x] = 0 ;
      //Serial.println("Weight:" + String(weights[y][x]));
    }
  }
}
//--------------------------------------------------------------------------------------------------------------
void show_weights() {

  Serial.println(F("Show weights:"));
  for ( int y = 0; y < total_fields; y++) {
    Serial.print("{");
    for ( int x = 0; x < total_fields; x++) {
      Serial.print(String(weights[y][x]) + " , ");
    }
    Serial.println("}");
  }
  Serial.println();
}
//--------------------------------------------------------------------------------------------------------------
void show_rewards() {

  Serial.println(F("Show rewards:"));
  for ( int y = 0; y < total_fields; y++) {
    Serial.print("{");
    for ( int x = 0; x < total_fields; x++) {
      Serial.print(String(rewards[y][x]) + " , ");
    }
    Serial.println("}");
  }
  Serial.println();
}
//--------------------------------------------------------------------------------------------------------------
void set_way_information() { //tft display

  int maximal_weight = 0;
  int copy_x = 0;

  set_field_information(start_field, 0);

  for ( int y = 0; y < total_fields; y++) {
    for ( int x = 0; x < total_fields; x++) {

      if (x == 0) {
        maximal_weight = 0;
        copy_x = 0;
      }

      if (weights[y][x] > maximal_weight) {
        maximal_weight = weights[y][x];
        copy_x = x;
      }

      if (x == (total_fields - 1)) {
        //Serial.println("Maximum Weight:" + String(maximal_weight) + "," + String(y) + "," + String(copy_x));
        set_field_information(copy_x, 0);
        y = copy_x - 1;
      }
    }
    if (maximal_weight == 100)break;
  }

  set_field_information(target_field, 0);
}
//--------------------------------------------------------------------------------------------------------------
void set_field_information(int field_number, int clear_old_field) { //display one field  n=0-7  ,tft display

  //Serial.println(F("Update Fields on TFT:"));
  int field_counter = 0;

  if ( clear_old_field == 1)clear_field_information(last_field);

  for ( int y = 0; y < culumns_of_fields; y++) {
    for ( int x = 0; x < lines_of_fields; x++) {
      if (field_counter == field_number) {
        SetFilledRect(WHITE , (x * size_fields), (y * size_fields), size_fields, size_fields);
        if (field_number == target_field)SetFilledRect(RED , (x * size_fields), (y * size_fields), size_fields, size_fields);
        if (field_number == start_field)SetFilledRect(YELLOW , (x * size_fields), (y * size_fields), size_fields, size_fields);
        SetRect(WHITE , (x * size_fields), (y * size_fields), size_fields, size_fields);
      }
      field_counter++;
    }
  }

  last_field = field_number;
}
//--------------------------------------------------------------------------------------------------------------
void clear_field_information(int field_number) { //clear one field  n=0-7 ,tft display

  //Serial.println(F("Clear Fields on TFT:"));
  int field_counter = 0;

  //display_field_frames();

  for ( int y = 0; y < culumns_of_fields; y++) {
    for ( int x = 0; x < lines_of_fields; x++) {
      if (field_counter == field_number) {
        SetFilledRect(BLACK , (x * size_fields), (y * size_fields), size_fields, size_fields);
        SetRect(WHITE , (x * size_fields), (y * size_fields), size_fields, size_fields);
      }
      field_counter++;
    }
  }
}
//--------------------------------------------------------------------------------------------------------------
void set_field_frames() { //tft display

  //Serial.println(F("Create Fields on TFT:"));

  for ( int y = 0; y < culumns_of_fields; y++) {
    for ( int x = 0; x < lines_of_fields; x++) {
      SetRect(WHITE , (x * size_fields), (y * size_fields), size_fields, size_fields);
    }
  }
}
//--------------------------------------------------------------------------------------------------------------
void ScreenText(uint16_t color, int xtpos, int ytpos, int text_size , String text) {
  tft.setCursor(xtpos, ytpos);
  tft.setTextColor(color);
  tft.setTextSize(text_size);
  tft.println(text);
}
//--------------------------------------------------------------------------------------------------------------
void SetLines(uint16_t color , int xl1pos, int yl1pos, int xl2pos, int yl2pos) {
  tft.drawLine(xl1pos, yl1pos, xl2pos, yl2pos, color);
}
//--------------------------------------------------------------------------------------------------------------
void SetPoint(uint16_t color, int xppos, int yppos) {
  tft.drawPixel(xppos, yppos, color);
}
//--------------------------------------------------------------------------------------------------------------
void SetRect(uint16_t color , int xr1pos, int yr1pos, int xr2width, int yr2hight) {
  tft.drawRect(xr1pos, yr1pos, xr2width, yr2hight, color);
}
//--------------------------------------------------------------------------------------------------------------
void SetFilledRect(uint16_t color , int xr1pos, int yr1pos, int xr2width, int yr2hight) {
  tft.fillRect(xr1pos, yr1pos, xr2width, yr2hight, color);
}
//--------------------------------------------------------------------------------------------------------------
void SetCircle(uint16_t color , int xcpos, int ycpos, int radius) {
  tft.drawCircle(xcpos, ycpos, radius, color);
}
//--------------------------------------------------------------------------------------------------------------
void SetFilledCircle(uint16_t color , int xcpos, int ycpos, int radius) {
  tft.fillCircle(xcpos, ycpos, radius, color);
}
