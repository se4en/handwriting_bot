from handwriting_synthesis.handwriting_synthesis import *

def handwrite(user_name, user_text):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #эти данные не меняются
    data_path = './synthesis/data/'
    model_path = './synthesis/best_model_synthesis.pt'

    save_file = './user_handwriting/' + user_name

    train_dataset = HandwritingDataset(data_path, split='train', text_req=True)

    #путь до папки пользователя, где сохранён его стиль (.txt и .npy)
    #@flask_app.route("/upload_style", methods=["GET", "POST"]) -> save_path - здесь отличается, надо откорректировать
    user_path = './app/user_data/'

    style = np.load(user_path + user_name + '.npy', allow_pickle=True, encoding='bytes').astype(np.float32)

    with open(user_path + user_name + '.txt') as file:
        texts = file.read().splitlines()
    real_text = texts[0]

    # plot_stroke(style)

    mean, std, style = data_normalization(style)
    style = torch.from_numpy(style).unsqueeze(0).to(device)
    gen_seq, phi = generate_sequence(
                model_path, user_text, device, train_dataset.char_to_id,
                train_dataset.idx_to_char, 8, True, style, real_text, False)
    end = style.shape[1]
    gen_seq = data_denormalization(mean, std, gen_seq)

    plot_stroke(gen_seq[0], save_file)
