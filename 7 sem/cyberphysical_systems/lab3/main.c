/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* Typical PCF8574 backpack: P0=RS, P1=RW, P2=EN, P3=backlight,
   P4..P7=D4..D7. */
#define LCD_RS         0x01U
#define LCD_EN         0x04U
#define LCD_BACKLIGHT  0x08U
#define BUTTON_DEBOUNCE_MS 30U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

/* USER CODE BEGIN PV */
/* Shifted address for STM32 HAL; determined automatically at startup. */
static uint16_t lcd_address;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Search PCF8574 (0x20..0x27) and PCF8574A (0x38..0x3F).
   Connect only one LCD backpack in these address ranges. */
static void LCD_FindAddress(void)
{
  for (uint8_t address = 0x20U; address <= 0x3FU; ++address)
  {
    if (address >= 0x28U && address < 0x38U)
      continue;

    if (HAL_I2C_IsDeviceReady(&hi2c1, (uint16_t)(address << 1),
                              2U, 20U) == HAL_OK)
    {
      lcd_address = (uint16_t)(address << 1);
      return;
    }
  }

  /* No response: check power, wiring and board switches. */
  Error_Handler();
}

static void LCD_WriteNibble(uint8_t nibble, uint8_t rs)
{
  uint8_t data = (nibble & 0xF0U) | LCD_BACKLIGHT | rs;
  uint8_t packet[3] = {data, (uint8_t)(data | LCD_EN), data};

  if (HAL_I2C_Master_Transmit(&hi2c1, lcd_address, packet,
                              sizeof(packet), 100U) != HAL_OK)
    Error_Handler();

  /* Covers ordinary LCD command execution time. */
  HAL_Delay(1U);
}

static void LCD_Send(uint8_t value, uint8_t rs)
{
  LCD_WriteNibble(value, rs);
  LCD_WriteNibble((uint8_t)(value << 4), rs);
}

static void LCD_Command(uint8_t command)
{
  LCD_Send(command, 0U);
  if (command == 0x01U || command == 0x02U)
    HAL_Delay(2U);
}

static void LCD_Init(void)
{
  HAL_Delay(50U);
  LCD_FindAddress();

  /* Force a known state even after an MCU-only reset. */
  LCD_WriteNibble(0x30U, 0U);
  HAL_Delay(5U);
  LCD_WriteNibble(0x30U, 0U);
  HAL_Delay(1U);
  LCD_WriteNibble(0x30U, 0U);
  LCD_WriteNibble(0x20U, 0U); /* 4-bit interface. */

  LCD_Command(0x28U);       /* 2 lines, 5x8 character font. */
  LCD_Command(0x08U);       /* Display off during setup. */
  LCD_Command(0x01U);       /* Clear display. */
  LCD_Command(0x06U);       /* Cursor advances right. */
  LCD_Command(0x0CU);       /* Display on, cursor off. */
}

static void LCD_Print(const char *text)
{
  while (*text != '\0')
    LCD_Send((uint8_t)*text++, LCD_RS);
}

static void LCD_ShowCounter(uint32_t value)
{
  char line[17];

  /* Always overwrite all 16 columns, including trailing spaces. */
  (void)snprintf(line, sizeof(line), "Count:%10lu", (unsigned long)value);
  LCD_Command(0x80U);       /* First line, column 0. */
  LCD_Print(line);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */
  uint32_t counter = 0U;
  uint32_t button_changed_at;
  GPIO_PinState button_raw;
  GPIO_PinState button_stable;

  LCD_Init();
  LCD_ShowCounter(counter);
  LCD_Command(0xC0U);       /* Second line, column 0. */
  LCD_Print("Press button    ");

  /* A button held during startup must first be released. */
  button_raw = HAL_GPIO_ReadPin(BUT_GPIO_Port, BUT_Pin);
  button_stable = button_raw;
  button_changed_at = HAL_GetTick();
/* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    GPIO_PinState sample = HAL_GPIO_ReadPin(BUT_GPIO_Port, BUT_Pin);
    uint32_t now = HAL_GetTick();

    if (sample != button_raw)
    {
      button_raw = sample;
      button_changed_at = now;
    }

    /* Accept a new state only after it stays unchanged for 30 ms. */
    if (button_raw != button_stable &&
        (uint32_t)(now - button_changed_at) >= BUTTON_DEBOUNCE_MS)
    {
      button_stable = button_raw;
      if (button_stable == GPIO_PIN_RESET)
      {
        ++counter;         /* One increment per press, no repeat on hold. */
        LCD_ShowCounter(counter);
      }
    }

    HAL_Delay(1U);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.Prediv1Source = RCC_PREDIV1_SOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  RCC_OscInitStruct.PLL2.PLL2State = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the Systick interrupt time
  */
  __HAL_RCC_PLLI2S_ENABLE();
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin : BUT_Pin */
  GPIO_InitStruct.Pin = BUT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(BUT_GPIO_Port, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
