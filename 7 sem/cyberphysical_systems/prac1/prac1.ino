// NUCLEO-C031C6: автомобильный и пешеходный светофор по кнопке.
// Пины и сценарий соответствуют методическим указаниям к практической №1.

#define LED_R 13   // машины: красный
#define LED_Y 12   // машины: жёлтый
#define LED_G 11   // машины: зелёный
#define LED_R_P 9  // пешеходы: красный
#define LED_G_P 8  // пешеходы: зелёный
#define BUT 4      // кнопка, подключённая к GND
#define POT A2     // потенциометр длительности пешеходной фазы
#define BUZZ A5    // зуммер

void cars(bool red, bool yellow, bool green) {
  digitalWrite(LED_R, red);
  digitalWrite(LED_Y, yellow);
  digitalWrite(LED_G, green);
}

void pedestrians(bool red, bool green) {
  digitalWrite(LED_R_P, red);
  digitalWrite(LED_G_P, green);
}

bool requestCross = false;

// Во время ожидания продолжаем опрашивать кнопку: запрос запоминается,
// но не прерывает текущую безопасную фазу машин.
void waitWithButton(uint32_t durationMs) {
  const uint16_t pollStepMs = 20;
  const uint32_t startedAt = millis();

  while (millis() - startedAt < durationMs) {
    if (digitalRead(BUT) == LOW) {
      delay(20); // антидребезг
      if (digitalRead(BUT) == LOW) {
        requestCross = true;
        while (digitalRead(BUT) == LOW) {
          delay(pollStepMs);
        }
      }
    }
    delay(pollStepMs);
  }
}

void pedestrianPhase(uint32_t durationMs) {
  cars(true, false, false);
  pedestrians(false, true);

  const uint32_t startedAt = millis();
  while (millis() - startedAt < durationMs) {
    const uint32_t elapsed = millis() - startedAt;
    const uint32_t remaining = durationMs > elapsed ? durationMs - elapsed : 0;

    // Короткий сигнал каждые 500 мс.
    if (elapsed % 500 < 120) {
      tone(BUZZ, 1000, 100);
    }

    // Последние две секунды разрешающий сигнал мигает.
    digitalWrite(LED_G_P, remaining <= 2000 ? (millis() / 200) % 2 : HIGH);
    delay(20);
  }

  pedestrians(true, false);
}

void setup() {
  pinMode(LED_R, OUTPUT);
  pinMode(LED_Y, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_R_P, OUTPUT);
  pinMode(LED_G_P, OUTPUT);
  pinMode(BUZZ, OUTPUT);
  pinMode(BUT, INPUT_PULLUP);
  pinMode(POT, INPUT);

  cars(false, false, true);
  pedestrians(true, false);
}

void loop() {
  // STM32 Nucleo-C031C6 возвращает 0..1023 в используемом шаблоне Wokwi.
  const uint32_t pedestrianTimeMs = map(analogRead(POT), 0, 1023, 3000, 10000);

  cars(false, false, true);
  pedestrians(true, false);
  waitWithButton(5000);

  cars(false, true, false);
  delay(1000);

  cars(true, false, false);
  delay(300);
  if (requestCross) {
    pedestrianPhase(pedestrianTimeMs);
    requestCross = false;
  } else {
    delay(2000);
  }

  cars(false, true, false);
  delay(800);
}
