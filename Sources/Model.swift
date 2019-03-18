import TensorFlow

public struct Linear: Layer {
    
    public var w: Tensor<Float>
    public var b: Tensor<Float>

    public init(inputSize: Int32, outputSize: Int32) {
        w = Tensor<Float>(zeros: [inputSize, outputSize])
        b = Tensor<Float>(zeros: [outputSize])
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float>{
        return matmul(input, w) + b
    }

}

public struct LinearSansBias: Layer {

    public var w: Tensor<Float>

    public init(inputSize: Int32, outputSize: Int32) {
        w = Tensor<Float>(zeros: [inputSize, outputSize])
    }

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float>{
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
    public func applied(to input: mLSTMInput, in context: Context) -> mLSTMHiddenState {
        let m = wmx.applied(to: input.data,in:context) * wmh.applied(to: input.hidden.h,in:context)

        let i = sigmoid(wxI.applied(to: input.data,in:context) + whI.applied(to: m,in:context))
        let f = sigmoid(wxF.applied(to: input.data,in:context) + whF.applied(to: m,in:context))
        let u = tanh(wxU.applied(to: input.data,in:context) + whU.applied(to: m,in:context))
        let o = sigmoid(wxO.applied(to: input.data,in:context) + whO.applied(to: m,in:context))

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
    public func applied(to input: mLSTMInput, in context: Context) -> StackedLSTMOutput{
        let rnnOutput = rnn.applied(to: input,in:context)
        let output = softmax(h2o.applied(to: rnnOutput.h,in: context))
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
    public func applied(to input: LanguageModellingInput, in context: Context) -> StackedLSTMOutput{
        let embedding = embed.applied(to: input.inputCharacter,in: context)
        return rnn.applied(to: mLSTMInput(data: embedding, hidden: input.hiddenState))
    }

}
