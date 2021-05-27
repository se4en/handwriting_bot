from handwriting_synthesis.handwriting_synthesis import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#эти данные не менятся
data_path = './data/'
model_path = './best_model_synthesis.pt'

train_dataset = HandwritingDataset(data_path, split='train', text_req=True)


# здесь должен быть текст, который пользователь ввёл в телеграмме
char_seq = "hello world!"

#путь до папки пользователя, где сохранён его стиль (.txt и .npy) 
#@flask_app.route("/upload_style", methods=["GET", "POST"]) -> save_path - здесь отличается, надо откорректировать
file_path = './styles/' + 'default/'

style = np.load(file_path + 'style.npy', allow_pickle=True, encoding='bytes').astype(np.float32)
with open(file_path + 'inpText.txt') as file:
    texts = file.read().splitlines()
real_text = texts[0]

plot_stroke(style)
mean, std, style = data_normalization(style)
style = torch.from_numpy(style).unsqueeze(0).to(device)
gen_seq, phi = generate_sequence(
            model_path, char_seq, device, train_dataset.char_to_id,
            train_dataset.idx_to_char, 8, True, style, real_text, False)
end = style.shape[1]
gen_seq = data_denormalization(mean, std, gen_seq)
plot_stroke(gen_seq[0])