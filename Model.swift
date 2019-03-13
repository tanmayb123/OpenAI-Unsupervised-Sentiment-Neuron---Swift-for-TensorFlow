public struct Linear: Layer {

    public var w: Tensor<Float>
    public var b: Tensor<Float>

    public init(inputSize: Int32, outputSize: Int32) {
        w = Tensor<Float>(zeros: [inputSize, outputSize])
        b = Tensor<Float>(zeros: [outputSize])
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return matmul(input, w) + b
    }

}

public struct LinearSansBias: Layer {

    public var w: Tensor<Float>

    public init(inputSize: Int32, outputSize: Int32) {
        w = Tensor<Float>(zeros: [inputSize, outputSize])
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return matmul(input, w)
    }

}

public struct mLSTMHiddenState: Differentiable {

    public var h: Tensor<Float>
    public var c: Tensor<Float>

    public init(h: Tensor<Float>, c: Tensor<Float>) {
        self.h = h
        self.c = c
    }

}

public struct mLSTMInput: Differentiable {

    public var data: Tensor<Float>
    public var hidden: mLSTMHiddenState

    public init(data: Tensor<Float>, hidden: mLSTMHiddenState) {
        self.data = data
        self.hidden = hidden
    }

}

public struct mLSTM: Layer {

    public var wxI: LinearSansBias
    public var wxF: LinearSansBias
    public var wxU: LinearSansBias
    public var wxO: LinearSansBias

    public var whI: Linear
    public var whF: Linear
    public var whU: Linear
    public var whO: Linear

    public var wmx: LinearSansBias
    public var wmh: LinearSansBias

    public init(inputSize: Int32, hiddenSize: Int32) {
        wxI = LinearSansBias(inputSize: inputSize, outputSize: hiddenSize)
        wxF = LinearSansBias(inputSize: inputSize, outputSize: hiddenSize)
        wxU = LinearSansBias(inputSize: inputSize, outputSize: hiddenSize)
        wxO = LinearSansBias(inputSize: inputSize, outputSize: hiddenSize)

        whI = Linear(inputSize: hiddenSize, outputSize: hiddenSize)
        whF = Linear(inputSize: hiddenSize, outputSize: hiddenSize)
        whU = Linear(inputSize: hiddenSize, outputSize: hiddenSize)
        whO = Linear(inputSize: hiddenSize, outputSize: hiddenSize)

        wmx = LinearSansBias(inputSize: inputSize, outputSize: hiddenSize)
        wmh = LinearSansBias(inputSize: hiddenSize, outputSize: hiddenSize)
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: mLSTMInput) -> mLSTMHiddenState {
        let m = wmx.applied(to: input.data) * wmh.applied(to: input.hidden.h)

        let i = sigmoid(wxI.applied(to: input.data) + whI.applied(to: m))
        let f = sigmoid(wxF.applied(to: input.data) + whF.applied(to: m))
        let u = tanh(wxU.applied(to: input.data) + whU.applied(to: m))
        let o = sigmoid(wxO.applied(to: input.data) + whO.applied(to: m))

        let cy = f * input.hidden.c + i * u
        let hy = o * tanh(cy)

        return mLSTMHiddenState(h: hy, c: cy)
    }

}

public struct StackedLSTMOutput: Differentiable {

    public var hidden: mLSTMHiddenState
    public var output: Tensor<Float>

    public init(hidden: mLSTMHiddenState, output: Tensor<Float>) {
        self.hidden = hidden
        self.output = output
    }

}

public struct StackedLSTM: Layer {

    public var rnn: mLSTM
    public var h2o: Linear

    public init() {
        rnn = mLSTM(inputSize: 64, hiddenSize: 4096)
        h2o = Linear(inputSize: 4096, outputSize: 256)
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: mLSTMInput) -> StackedLSTMOutput {
        let rnnOutput = rnn.applied(to: input)
        let output = softmax(h2o.applied(to: rnnOutput.h))
        return StackedLSTMOutput(hidden: mLSTMHiddenState(h: rnnOutput.h, c: rnnOutput.c), output: output)
    }

    public func initialState(batchSize: Int32) -> mLSTMHiddenState {
        return mLSTMHiddenState(h: Tensor<Float>(zeros: [batchSize, 4096]), c: Tensor<Float>(zeros: [batchSize, 4096]))
    }

}

public struct LanguageModellingInput: Differentiable {

    public var inputCharacter: Tensor<Float>
    public var hiddenState: mLSTMHiddenState

    public init(inputCharacter: Tensor<Float>, hiddenState: mLSTMHiddenState) {
        self.inputCharacter = inputCharacter
        self.hiddenState = hiddenState
    }

}

public struct LanguageModelling: Layer {

    public var embed: LinearSansBias
    public var rnn: StackedLSTM

    public init() {
        embed = LinearSansBias(inputSize: 256, outputSize: 64)
        rnn = StackedLSTM()
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: LanguageModellingInput) -> StackedLSTMOutput {
        let embedding = embed.applied(to: input.inputCharacter)
        return rnn.applied(to: mLSTMInput(data: embedding, hidden: input.hiddenState))
    }

}
