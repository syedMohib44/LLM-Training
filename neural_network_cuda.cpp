#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iterator>

// Define the TinyLanguageModel class
struct TinyLanguageModel : torch::nn::Module
{
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};

    TinyLanguageModel(int vocab_size, int embedding_dim, int hidden_dim)
        : embedding(register_module("embedding", torch::nn::Embedding(vocab_size, embedding_dim))),
          lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embedding_dim, hidden_dim).batch_first(true)))),
          fc(register_module("fc", torch::nn::Linear(hidden_dim, vocab_size))) {}

    torch::Tensor forward(torch::Tensor x)
    {
        auto embeddings = embedding(x);
        auto lstm_out = std::get<0>(lstm(embeddings));
        auto output = fc(lstm_out.select(1, lstm_out.size(1) - 1)); // Use the last output of the LSTM
        return output;
    }
};

// Utility functions for encoding/decoding words
std::unordered_map<std::string, int> build_vocab(const std::vector<std::string> &corpus)
{
    std::unordered_map<std::string, int> word_to_idx;
    int idx = 0;
    for (const auto &sentence : corpus)
    {
        std::istringstream stream(sentence);
        std::string word;
        while (stream >> word)
        {
            /**
             * word_to_idx.find(word):
             * The find() function in std::unordered_map attempts to locate the key word in the map
             * If the key is found, it returns an iterator pointing to the key-value pair where the key is located
             * If the key is not found, it returns word_to_idx.end()
             * word_to_idx.end()
             * end() is a sentinel iterator that does not point to any valid element. It marks the end of the map's elements
             * If find() returns end(), it means the key was not found
             
             * The Condition
             * if (word_to_idx.find(word) == word_to_idx.end()
             
             * This checks whether the iterator returned by find(word) is equal to end()
             * If it is, the key word does not exist in the map.
             */
            if (word_to_idx.find(word) == word_to_idx.end())
            {
                std::cout << "&&&& " << word << "\n"; 
                word_to_idx[word] = idx++;
            }
        }
    }
    return word_to_idx;
}

std::vector<std::pair<std::vector<int>, int> > create_dataset(const std::vector<std::string> &corpus,
                                                              const std::unordered_map<std::string, int> &word_to_idx,
                                                              int sequence_length)
{
    std::vector<std::pair<std::vector<int>, int> > dataset;
    for (const auto &sentence : corpus)
    {
        std::istringstream stream(sentence);
        /**
         * std::istream_iterator<std::string>{stream}
         * This creates an iterator that reads from the stream.
         * std::istream_iterator<std::string> reads space-separated words (or, in general, elements of the type specified in angle brackets, here std::string) from a stream.
         * The first {stream} initializes the iterator at the beginning of the stream.
         * std::istream_iterator<std::string>{}
         * This creates an end-of-stream iterator. It acts as a "sentinel" indicating the end of the input.
         * std::vector Initialization
         * std::vector supports a constructor that takes a pair of iterators as arguments.
         * std::istream_iterator<std::string>{stream} is the beginning of the range.
         * std::istream_iterator<std::string>{} is the end of the range.
         * The std::vector constructor reads words from stream (separated by whitespace) and stores them in the tokens vector until the end of the stream is reached. 
         */

        /**
         * Example:
         * The std::istringstream is initialized with "Hello world this is C++".
         * std::istream_iterator<std::string>{stream} reads words one by one:
         * First word: "Hello"
         * Second word: "world"
         * Third word: "this"
         * Fourth word: "is"
         * Fifth word: "C++"
         */
        std::cout << sentence << "\n";
        std::vector<std::string> tokens{std::istream_iterator<std::string>{stream}, std::istream_iterator<std::string>{}};
        for (size_t i = 0; i + sequence_length < tokens.size(); ++i)
        {
            std::vector<int> x(sequence_length);
            for (size_t j = 0; j < sequence_length; ++j)
            {
                x[j] = word_to_idx.at(tokens[i + j]);
            }
            int y = word_to_idx.at(tokens[i + sequence_length]);
            std::cout << " === " << tokens[i + sequence_length] << "\n";
            dataset.emplace_back(x, y);
        }
    }
    return dataset;
}

// Main function
int main()
{
    // Corpus and parameters
    std::vector<std::string> corpus = {
        "the sky is blue",
        "the grass is green",
        "the sun is bright",
        "the moon is white",
    };

    // // Build vocabulary
    // auto word_to_idx = build_vocab(corpus);
    // std::cout << "Vocabulary:\n";
    // for (const auto &pair : word_to_idx)
    // {
    //     std::cout << pair.first << " -> " << pair.second << "\n";
    // }

    // // Sequence length
    // int sequence_length = 3;

    // // Create dataset
    // std::vector<std::pair<std::vector<int>, int> > dataset = create_dataset(corpus, word_to_idx, sequence_length);
    // std::cout << "\nDataset:\n";
    // for (const std::pair<std::vector<int>, int> &pair : dataset)
    // {
    //     const std::vector<int> &x = pair.first; // Extract the input vector
    //     std::cout << "Input: [ ";
    //     for (const int &val : x)
    //     {
    //         std::cout << val << " ";
    //     }
    //     int y = pair.second; // Extract the output value
    //     std::cout << "] -> Output: " << y << "\n";
    // }

    int sequence_length = 3;
    int embedding_dim = 10;
    int hidden_dim = 20;
    int epochs = 200;
    double learning_rate = 0.01;

    // Build vocabulary and dataset
    auto word_to_idx = build_vocab(corpus);
    auto idx_to_word = [&word_to_idx]()
    {
        std::unordered_map<int, std::string> idx_to_word;
        for (const auto &pair : word_to_idx)
        {
            idx_to_word[pair.second] = pair.first;
        }
        return idx_to_word;
    }();
    auto dataset = create_dataset(corpus, word_to_idx, sequence_length);

    // Define the model
    int vocab_size = word_to_idx.size();
    TinyLanguageModel model(vocab_size, embedding_dim, hidden_dim);
    model.to(torch::kCPU);

    // Loss and optimizer
    auto criterion = torch::nn::CrossEntropyLoss();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;
        for (const auto &[x, y] : dataset)
        {
            auto inputs = torch::tensor(x, torch::dtype(torch::kLong)).unsqueeze(0); // Batch size = 1
            auto target = torch::tensor(y, torch::dtype(torch::kLong));

            optimizer.zero_grad();
            auto output = model.forward(inputs);
            auto loss = criterion(output, target.unsqueeze(0));
            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>();
        }
        if ((epoch + 1) % 20 == 0)
        {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss << std::endl;
        }
    }

    // Test the model
    std::vector<std::string> test_input = {"the", "sky", "is"};
    std::vector<int> test_encoded;
    for (const auto &word : test_input)
    {
        test_encoded.push_back(word_to_idx[word]);
    }

    auto test_tensor = torch::tensor(test_encoded, torch::dtype(torch::kLong)).unsqueeze(0);
    auto output = model.forward(test_tensor);
    int predicted_idx = output.argmax(1).item<int>();

    std::cout << "Input: ";
    for (const auto &word : test_input)
    {
        std::cout << word << " ";
    }
    std::cout << "\nPredicted next word: " << idx_to_word.at(predicted_idx) << std::endl;

    return 0;
}