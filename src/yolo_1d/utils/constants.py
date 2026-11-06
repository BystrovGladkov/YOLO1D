# Значения по умолчанию для исходной (вихретоковой) конфигурации.
# Для других данных длины задаются в настройках: AugmentationSettings.regular_length / input_length,
# число каналов — TrainingSettings.input_channels (ЭКГ LUDB: 5000 / 5120 / 1).

# length on one signal sample
REGULAR_LENGTH = 50000
# length of network input
INPUT_LENGTH = 49152
