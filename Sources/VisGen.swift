import Python

public class TextGenModel {

    public var languageModeller = LanguageModelling()
    public var currentHiddenState: mLSTMHiddenState

    public init() {
        currentHiddenState = languageModeller.rnn.initialState(batchSize: 1)
        loadWeights()
    }

    private func loadWeights() {
        languageModeller.embed.w = Tensor<Float>(numpy: np.load("weights/embd.npy"))!

        languageModeller.rnn.h2o.w = Tensor<Float>(numpy: np.load("weights/w.npy"))!
        languageModeller.rnn.h2o.b = Tensor<Float>(numpy: np.load("weights/b.npy"))!

        let wx = np.split(np.load("weights/wx.npy"), 4, axis: 1).map { numpyArray -> Tensor<Float> in
            return Tensor<Float>(numpy: numpyArray)!
        }
        languageModeller.rnn.rnn.wxI.w = wx[0]
        languageModeller.rnn.rnn.wxF.w = wx[1]
        languageModeller.rnn.rnn.wxO.w = wx[2]
        languageModeller.rnn.rnn.wxU.w = wx[3]

        let wh = np.split(np.load("weights/wh.npy"), 4, axis: 1).map { numpyArray -> Tensor<Float> in
            return Tensor<Float>(numpy: numpyArray)!
        }
        languageModeller.rnn.rnn.whI.w = wh[0]
        languageModeller.rnn.rnn.whF.w = wh[1]
        languageModeller.rnn.rnn.whO.w = wh[2]
        languageModeller.rnn.rnn.whU.w = wh[3]

        let b0 = np.split(np.load("weights/b0.npy"), 4, axis: 0).map { numpyArray -> Tensor<Float> in
            return Tensor<Float>(numpy: numpyArray)!
        }
        languageModeller.rnn.rnn.whI.b = b0[0]
        languageModeller.rnn.rnn.whF.b = b0[1]
        languageModeller.rnn.rnn.whO.b = b0[2]
        languageModeller.rnn.rnn.whU.b = b0[3]

        languageModeller.rnn.rnn.wmx.w = Tensor<Float>(numpy: np.load("weights/wmx.npy"))!

        languageModeller.rnn.rnn.wmh.w = Tensor<Float>(numpy: np.load("weights/wmh.npy"))!
    }

    public func applied(to input: String, withTemperature temperature: Float,in:Context) -> String {
        let ordinal = Int(Python.ord(Python.str(input).encode()))!
        var oneHotInput = [Int](repeating: 0, count: 256)
        oneHotInput[ordinal] = 1

        let predicted = languageModeller.applied(to: LanguageModellingInput(inputCharacter: Tensor<Float>(numpy: np.expand_dims(np.array(oneHotInput, dtype: "float32"), axis: 0))!, hiddenState: currentHiddenState))
        currentHiddenState = predicted.hidden

        if temperature == -1 {
            return ""
        }

        var prediction = predicted.output.makeNumpyArray().astype("float64")[0]

        if temperature == 0 {
            let predictedCharacter = np.argmax(prediction)
            return String(Python.chr(predictedCharacter))!
        }

        prediction = np.log(prediction) / PythonObject(temperature)
        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)

        let predictedCharacter = np.argmax(np.random.multinomial(1, prediction, 1))
        return String(Python.chr(predictedCharacter))!
    }

}

func renderTextWithRGB(text: [String], withRGB: [[Int]], desiredWidth: Int) -> PythonObject {
    func getCharacterWidthPadding(character: String) -> (Int, Int) {
        return character == " " ? (7, 0) : (5, 2)
    }

    let initialX = 10
    let initialY = 10
    let initialCharWidthPadding = getCharacterWidthPadding(character: text[0])
    var currentX = initialX - initialCharWidthPadding.0
    var currentY = initialY - 5
    var renderAt: [[Int]] = []

    for char in text {
        let (charWidth, charPadding) = getCharacterWidthPadding(character: char)
        if currentX + charWidth > desiredWidth - initialX {
            currentX = initialX + charPadding
            currentY += 12
        } else {
            currentX += charWidth + charPadding
        }
        renderAt.append([currentX, currentY])
    }

    let imageHeight = currentY + initialY + 7

    let image = Image.new("RGB", Python.tuple([desiredWidth, imageHeight]), color: Python.tuple([255, 255, 255]))
    let drawer = ImageDraw.Draw(image)

    for char in Python.list(Python.zip(text, renderAt, withRGB)) {
        let (character, renderLocation, renderColor) = (char[0], char[1], char[2])
        drawer.text(Python.tuple(renderLocation), character, fill: Python.tuple(renderColor))
    }

    return image
}

func renderStringWithNeuronColor(text: [String], neuronValues: [Float], desiredWidth: Int, neuronRange: Float = 0.8) -> PythonObject {
    let colors: [[Int]] = neuronValues.map { neuronValue -> [Int] in
        let green = Int((Float((max(-neuronRange, min(neuronValue, neuronRange)) + neuronRange) / Float(neuronRange * 2))) * 255.0)
        let red = 255 - green
        if green > red {
            return [0, green, 0]
        } else if red > green {
            return [red, 0, 0]
        } else {
            return [red, green, 0]
        }
    }

    return renderTextWithRGB(text: text, withRGB: colors, desiredWidth: desiredWidth)
}
