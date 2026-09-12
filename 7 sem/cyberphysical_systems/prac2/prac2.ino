#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

// Nucleo-C031C6 + ILI9341, два потенциометра и кнопка.
#define POT_Y A2
#define POT_X A3
#define BUTTON_PIN 4
#define TFT_DC 2
#define TFT_CS 3

Adafruit_ILI9341 tft(TFT_CS, TFT_DC);

const char LABEL[] = "RTU MIREA";
bool alternateStyle = false;
int16_t previousX = -1;
int16_t previousY = -1;
uint8_t previousSize = 0;
uint32_t lastFrameAt = 0;

uint16_t textColor() {
  return alternateStyle ? ILI9341_YELLOW : ILI9341_GREEN;
}

uint8_t textSize() {
  return alternateStyle ? 2 : 3;
}

int16_t labelWidth(uint8_t size) {
  return (sizeof(LABEL) - 1) * 6 * size;
}

int16_t labelHeight(uint8_t size) {
  return 8 * size;
}

void eraseObject(int16_t x, int16_t y, uint8_t size) {
  // Стираем только область старой надписи и рамки, а не весь дисплей.
  tft.fillRect(x - 2, y - 2, labelWidth(size) + 5, labelHeight(size) + 5, ILI9341_BLACK);
}

void drawObject(int16_t x, int16_t y, uint8_t size, uint16_t color) {
  tft.drawRect(x - 1, y - 1, labelWidth(size) + 2, labelHeight(size) + 2, color);
  tft.setCursor(x, y);
  tft.setTextColor(color);
  tft.setTextSize(size);
  tft.print(LABEL);
}

bool buttonPressed() {
  static bool previousRaw = HIGH;
  static bool stableState = HIGH;
  static uint32_t lastChangeAt = 0;

  const bool raw = digitalRead(BUTTON_PIN);
  if (raw != previousRaw) {
    previousRaw = raw;
    lastChangeAt = millis();
  }

  if (millis() - lastChangeAt >= 25 && raw != stableState) {
    stableState = raw;
    return stableState == LOW;
  }
  return false;
}

void setup() {
  pinMode(POT_X, INPUT);
  pinMode(POT_Y, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  tft.begin();
  tft.setRotation(1); // альбомный экран: 320x240
  tft.fillScreen(ILI9341_BLACK);
}

void loop() {
  // Ограничение частоты обновления: максимум 20 кадров в секунду.
  if (millis() - lastFrameAt < 50) {
    return;
  }
  lastFrameAt = millis();

  bool styleChanged = false;
  if (buttonPressed()) {
    alternateStyle = !alternateStyle;
    styleChanged = true;
  }

  const uint8_t size = textSize();
  const int16_t width = labelWidth(size);
  const int16_t height = labelHeight(size);
  const int16_t x = map(analogRead(POT_X), 0, 1023, 2, tft.width() - width - 2);
  const int16_t y = map(analogRead(POT_Y), 0, 1023, 2, tft.height() - height - 2);

  if (x != previousX || y != previousY || size != previousSize || styleChanged) {
    if (previousX >= 0) {
      eraseObject(previousX, previousY, previousSize);
    }
    drawObject(x, y, size, textColor());
    previousX = x;
    previousY = y;
    previousSize = size;
  }
}
