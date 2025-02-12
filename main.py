import logging
from flask import Flask, render_template, request, jsonify
import torch
from torch.utils.data import Dataset, DataLoader
import os
from markupsafe import escape
from tqdm import tqdm

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),  # Логи сохраняются в файл (кодировка UTF-8)
        logging.StreamHandler()                            # Логи выводятся в консоль
    ]
)
logger = logging.getLogger(__name__)

# Глобальные параметры
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
SEQ_LENGTH = 10  # Длина входной последовательности
BATCH_SIZE = 32
MODEL_PATH = "lstm_language_model.pth"

app = Flask(__name__)

# Загрузка модели
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Берем выход для последнего токена
        return output


def load_or_train_model(file_path="your_dataset.txt"):
    """
    Загружает существующую модель или обучает новую при необходимости.
    """
    if os.path.exists(MODEL_PATH):
        logger.info("Обнаружена предобученная модель.")
        user_input = input("Использовать предобученную модель? (y/n): ").strip().lower()
        if user_input == 'y':
            logger.info("Загрузка предобученной модели...")
            model, vocab, reverse_vocab = load_model()
            return model, vocab, reverse_vocab

    logger.info("Начало обучения новой модели...")
    text = load_text_data(file_path)
    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)
    vocab = {char: i for i, char in enumerate(chars)}
    reverse_vocab = {i: char for i, char in enumerate(chars)}

    dataset = CustomTextDataset(text, vocab, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, optimizer, criterion, epochs=3, device=device)

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'reverse_vocab': reverse_vocab
    }, MODEL_PATH)

    logger.info(f"Модель успешно сохранена в {MODEL_PATH}.")
    return model, vocab, reverse_vocab


def load_model():
    """
    Загружает сохраненную модель и словари.
    """
    checkpoint = torch.load(MODEL_PATH)
    chars = sorted(list(checkpoint['vocab'].keys()))
    VOCAB_SIZE = len(chars)
    model = LSTMModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    reverse_vocab = checkpoint['reverse_vocab']
    return model, vocab, reverse_vocab


def load_text_data(file_path):
    """Загружает данные из текстового файла."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        logger.warning(f"Файл {file_path} не найден. Использование тестовой последовательности.")
        text = "this is an example of a simple language model" * 1000
    return text


class CustomTextDataset(Dataset):
    """Набор данных для обработки текстовых последовательностей."""

    def __init__(self, text, vocab, seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        self.inputs, self.targets = self.create_sequences(text)

    def create_sequences(self, text):
        inputs, targets = [], []
        for i in range(len(text) - self.seq_length):
            seq_in = text[i:i + self.seq_length]
            seq_out = text[i + self.seq_length]
            inputs.append([self.vocab[char] for char in seq_in])
            targets.append(self.vocab[seq_out])
        return torch.tensor(inputs), torch.tensor(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_model(model, dataloader, optimizer, criterion, epochs, device):
    """Обучение модели."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def generate_text(model, vocab, reverse_vocab, seed_text, length=100, temperature=1.0, device="cpu"):
    """Генерация текста на основе начального текста."""
    model.eval()
    generated = list(seed_text)
    input_seq = [vocab[char] for char in seed_text][-SEQ_LENGTH:]
    with torch.no_grad():
        for _ in range(length):
            input_tensor = torch.tensor([input_seq]).to(device)
            output = model(input_tensor)
            output_dist = torch.softmax(output / temperature, dim=1)
            predicted_token = torch.multinomial(output_dist, 1).item()
            generated.append(reverse_vocab[predicted_token])
            input_seq.append(predicted_token)
            input_seq = input_seq[-SEQ_LENGTH:]  # Обновляем входную последовательность
    return ''.join(generated)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Главный маршрут для обработки GET и POST запросов.
    Обрабатывает действия кнопок и отображает результаты.
    """
    text1 = ""
    text2 = ""
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    logger.info(f"Получен запрос от IP: {client_ip}, User-Agent: {user_agent}")

    if request.method == 'POST':
        try:
            text1 = escape(request.form.get('text1', "").strip())  # Очистка и валидация входных данных
            if not text1:
                text2 = "Введите текст!"
                logger.warning("Обнаружен пустой ввод.")
                return render_template('index.html', text1=text1, text2=text2)

            # Генерация текста с использованием модели
            seed_text = text1
            generated_text = generate_text_with_model(seed_text)
            text2 = generated_text

        except Exception as e:
            logger.error(f"Непредвиденная ошибка в маршруте index: {e}")
            text2 = "Произошла непредвиденная ошибка. Пожалуйста, попробуйте снова."

    return render_template('index.html', text1=text1, text2=text2)


def generate_text_with_model(seed_text):
    """
    Генерирует текст с использованием предобученной модели.
    """
    global model, vocab, reverse_vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_text = generate_text(model, vocab, reverse_vocab, seed_text, length=50, temperature=0.8, device=device)
    logger.info(f"Сгенерированный текст: {generated_text}")
    return generated_text


if __name__ == '__main__':
    # Загрузка или обучение модели
    model, vocab, reverse_vocab = load_or_train_model()

    # Запуск Flask-приложения
    logger.info("Запуск Flask-приложения.")
    app.run(debug=True)