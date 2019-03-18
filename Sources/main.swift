import TensorFlow

//enableGPU()

import Python

let model = TextGenModel()

let seedText = CommandLine.arguments[1].map { char -> String in
    return "\(char)"
}
let seqLength = Int32(CommandLine.arguments[2])!
let temperature = Float(CommandLine.arguments[3])!
let neuron = Int32(CommandLine.arguments[4])!
let overwrite = Float(CommandLine.arguments[5])!
let imageFilename = CommandLine.arguments[6]

if seqLength == -1 && neuron == -1 {
    print("ERROR: Neither generating nor visualizing.")
} else {
    var neuronStates: [Float] = []
    var lastGeneratedString = ""
    for char in seedText {
        lastGeneratedString = model.applied(to: char, withTemperature: temperature)
        if neuron != -1 {
            neuronStates.append(model.currentHiddenState.h[0][neuron].scalarized())
        }
    }

    if seqLength != -1 {
        var generatedString = ["\(lastGeneratedString)"]
        for _ in 1...seqLength {
            if neuron != -1 && overwrite != -0.0012 {
                model.currentHiddenState.h[0][neuron] = Tensor<Float>(overwrite)
            }
            generatedString.append(model.applied(to: generatedString.last!, withTemperature: temperature))
            if neuron != -1 {
                neuronStates.append(model.currentHiddenState.h[0][neuron].scalarized())
            }
        }

        if neuron != -1 {
            renderStringWithNeuronColor(text: seedText + generatedString, neuronValues: neuronStates, desiredWidth: 500).save(imageFilename)
        } else {
            renderTextWithRGB(text: seedText + generatedString, withRGB: [[Int]](repeating: [0, 0, 0], count: seedText.count + generatedString.count), desiredWidth: 500).save(imageFilename)
        }
    } else {
        renderStringWithNeuronColor(text: seedText, neuronValues: neuronStates, desiredWidth: 500).save(imageFilename)
    }
}
