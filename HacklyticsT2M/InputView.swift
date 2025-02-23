//
//  InputView.swift
//  HacklyticsT2M
//
//  Created by Степан Кравцов on 2/22/25.
//

import SwiftUI
import Giffy

struct ModelLoadingView: View {
    let messages: [String] = ["Drink water before meals to control portion size.",
                              "Take the stairs instead of elevator when possible.",
                              "Stretch for five minutes after waking up daily.",
                              "Practice proper form over lifting heavy weights.",
                              "Keep a consistent sleep schedule for better recovery.",
                              "Add protein to every meal you eat.",
                              "Take regular breaks from sitting to move around.",
                              "Track your workouts to monitor your progress.",
                              "Focus on compound exercises for maximum muscle gain.",
                              "Stay hydrated throughout your workout session.",
                              "Remember to breathe properly during exercise sets.",
                              "Warm up properly before intense physical activity.",
                              "Listen to your body and rest when needed.",
                              "Mix cardio and strength training for best results.",
                              "Plan your meals ahead for healthier choices.",
                              "Maintain good posture throughout the day.",
                              "Get at least seven hours of sleep nightly.",
                              "Include vegetables in at least two daily meals.",
                              "Start with bodyweight exercises to build foundation.",
                              "Schedule your workouts like important meetings.",
                              "Focus on gradual progress, not instant results.",
                              "Keep healthy snacks readily available at home.",
                              "Exercise with a friend for better motivation.",
                              "Meal prep on weekends for healthy weekday eating.",
                              "Engage your core in everyday activities.",
                              "Take rest days to allow proper recovery.",
                              "Do dynamic stretches before working out.",
                              "Stay consistent with your workout routine.",
                              "Choose whole foods over processed alternatives.",
                              "Practice mindful eating during every meal.",
                              "Walk for at least thirty minutes each day.",
                              "Include flexibility training in your weekly routine.",
                              "Drink water first thing in the morning.",
                              "Focus on quality movement over exercise quantity.",
                              "Keep a workout journal to track progress.",
                              "Do bodyweight exercises during work breaks.",
                              "Practice proper breathing during strength training.",
                              "Fuel your body before morning workouts.",
                              "Include recovery exercises in your routine.",
                              "Stand up and move every hour.",
                              "Learn proper form before increasing weights.",
                              "Stay active on rest days with light walking.",
                              "Incorporate balance exercises into your routine.",
                              "Eat slowly and mindfully at every meal.",
                              "Challenge yourself with new exercises regularly.",
                              "Keep a water bottle within reach always.",
                              "Include protein in your post-workout meal.",
                              "Stretch major muscle groups daily for flexibility.",
                              "Plan active breaks during your workday.",
                              "Practice good form during everyday movements."]
    @State var currentMessage: String = "Thinking hard..."
    let timer = Timer.publish(every: 5, on: .current, in: .common).autoconnect()
    @State private var downloadAmount = 0.0
    let downloadTimer = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack {
            Text(currentMessage)
                .font(.subheadline)
                .foregroundColor(.white)
                .padding()
                .onReceive(timer) {_ in
                    if downloadAmount > 190 {
                        currentMessage = "Almost there..."
                    } else {
                        currentMessage = messages[Int.random(in: 0..<messages.count)]
                    }
                }
            ProgressView(value: downloadAmount, total: 220)
                .onReceive(downloadTimer) { _ in
                                if downloadAmount < 220 {
                                    downloadAmount += 1
                                }
                            }
                .padding()
        }
    }
}



struct InputView: View {
    @StateObject var speechRecognizer = SpeechRecognizer()
    @State var isRecording = false
    @State private var drawingHeight = true
    @State private var phase = 0.0
    @State private var imageData: Data?
    @State var isLoading: Bool = false
    @State var isShowingGif = false
    @State var wasError: Bool = false
    @State private var historyIDs: [String] = UserDefaults.standard.array(forKey: "historyIDs") as? [String] ?? []
    @State private var historyQuestions: [String] = UserDefaults.standard.array(forKey: "historyQuestions") as? [String] ?? []
    @State private var historyDates: [String] = UserDefaults.standard.array(forKey: "historyDates") as? [String] ?? []
    @State var animationID: String = ""
    @State var question: String = ""
    @State var currDate: String = ""
    var body: some View {
        VStack {
            if !isLoading {
                if !isShowingGif {
                    ZStack {
                        VStack {
                            Text("Describe your problem")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                                .padding(.top, 30)
                            
                            

                            
                            
                            if isRecording {
                                // Recording UI
                                VStack {
                                    Text("Recording in progress...")
                                        .font(.headline)
                                        .foregroundColor(.indigo)
                                        .padding()
                                    
                                    ZStack {
                                        ForEach(0..<8) { i in
                                            Wave(strength: 50, frequency: 10, phase: self.phase)
                                                .stroke(Color.white.opacity(Double(i) / 20), lineWidth: 5)
                                                .offset(y: CGFloat(i) * 5)
                                            
                                        }
                                    }
                                    .onAppear {
                                        withAnimation(Animation.linear(duration: 1).repeatForever(autoreverses: false)) {
                                            self.phase = .pi * 2
                                        }
                                    }
                                    
                                    
                                }
                            } else if !speechRecognizer.transcript.isEmpty {
                                // Final transcript display
                                VStack(spacing: 15) {
                                    Text("Transcript:")
                                        .font(.headline)
                                        .frame(alignment: .leading)
                                    
                                    ScrollView {
                                        Text(speechRecognizer.transcript)
                                            .font(.body)
                                            .padding()
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .background(Color(.systemGray6))
                                            .cornerRadius(10)
                                    }
                                    Button(action: {
                                        Task {
                                            DispatchQueue.main.async {
                                                isLoading = true
                                                wasError = false
                                            }
                                            let (animationID, question) = await getAnimationID(for: speechRecognizer.transcript)
                                            if animationID == "error" {
                                                DispatchQueue.main.async {
                                                    speechRecognizer.resetTranscript()
                                                    wasError = true
                                                    isLoading = false
                                                }
                                                return
                                            }
                                            
                                            let imageData = await getAnimation(for: animationID)
                                            let currentDate = Date()
                                            let dateString = currentDate.formatted(date: .numeric, time: .standard)
                                            DispatchQueue.main.async {
                                                self.imageData = imageData
                                                self.animationID = animationID
                                                self.question = question
                                                self.currDate = dateString
                                                isLoading = false
                                                isShowingGif = true
                                            }
                                            
                                        }
                                    }) {
                                        Text("Create motion")
                                            .foregroundStyle(.white)
                                            .padding()
                                            .background {
                                                RoundedRectangle(cornerRadius: 16)
                                                    .foregroundStyle(.indigo.opacity(0.6))
                                            }
                                    }
                                }
                                .padding()
                            } else {
                                // Initial state
                                VStack {
                                    
                                    Text("Tap the button below to start recording")
                                        .font(.headline)
                                        .foregroundColor(.secondary)
                                        .padding()
                                    if wasError {
                                        Text("There was an error with your request. Please try again.")
                                            .font(.headline)
                                            .foregroundColor(.red)
                                            .padding()
                                            .multilineTextAlignment(.center)
                                    }
                                }
                            }
                            
                            Spacer()
                            
                            RecordButton(isRecording: $isRecording) {
                                speechRecognizer.resetTranscript()
                                speechRecognizer.startTranscribing()
                                
                            } stopAction: {
                                speechRecognizer.stopTranscribing()
                                phase = 0
                                
                                
                            }
                            .frame(width: 70, height: 70)
                            
                            .padding(.bottom, 30)
                        }
                        NavigationLink(destination: HistoryView(historyIDs: $historyIDs, historyQuestions: $historyQuestions, dates: $historyDates)) {
                            Image(systemName: "folder.fill")
                                .font(.system(size: 18))
                                .foregroundStyle(.white)
                                .frame(width: 24, height: 24)
                                .padding(12)
                            
                                .background {
                                    ZStack {
                                        Circle()
                                            .foregroundStyle(.indigo.opacity(0.6))
                                        Circle()
                                            .stroke(.white.opacity(0.6), lineWidth: 1)
                                    }
                                        
                                }
                        }
                        .padding()
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                    }
                } else {
                    Text("Recommended motion")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .padding(.top, 30)
                    
                    ZStack {
                        if imageData != nil {
                            Giffy(imageData: imageData!)
                                .frame(maxWidth: .infinity)
                                .offset(y: -50)
                            
                        }
                        VStack {
                            Spacer()
                            ScrollView {
                                Text(LocalizedStringKey(question))
                                    .font(.body)
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    
                            }
                            .frame(maxWidth: .infinity, maxHeight: 300)
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                        }
                    }
                    
                    HStack {
                        Button(action: {
                            speechRecognizer.resetTranscript()
                            isShowingGif = false
                        }) {
                            Text("Back")
                                .foregroundStyle(.white)
                                .padding(12)
                                .background {
                                    RoundedRectangle(cornerRadius: 16)
                                        .foregroundStyle(.red.opacity(0.6))
                                }
                        }
                        Spacer()
                            .frame(width: 32)
                        Button(action: {
                            historyIDs.append(animationID)
                            historyQuestions.append(question)
                            historyDates.append(currDate)
                            UserDefaults.standard.set(historyIDs, forKey: "historyIDs")
                            UserDefaults.standard.set(historyQuestions, forKey: "historyQuestions")
                            UserDefaults.standard.set(historyDates, forKey: "historyDates")
                            speechRecognizer.resetTranscript()
                            isShowingGif = false
                        }) {
                            Text("Save")
                                .foregroundStyle(.white)
                                .padding(12)
                                .background {
                                    RoundedRectangle(cornerRadius: 16)
                                        .foregroundStyle(.indigo.opacity(0.6))
                                }
                        }
                    }
                }
            } else {
                ModelLoadingView()
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical)
//        .onAppear {
//            let domain = Bundle.main.bundleIdentifier!
//            UserDefaults.standard.removePersistentDomain(forName: domain)
//            UserDefaults.standard.synchronize()
//            print(historyIDs)
//            print(historyQuestions)
//            print(historyDates)
//        }
        
        
    }
    func getAnimationID(for prompt: String) async -> (String, String) {
        let url = URL(string: "https://e025-128-61-3-180.ngrok-free.app/get_exercise")!
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpMethod = "POST"
        let parameters: [String: Any] = [
            "question": prompt
        ]
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: parameters, options: .prettyPrinted)
        } catch let error {
            print(error.localizedDescription)
            return ("error", "error")
        }
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            if let jsonResponse = try JSONSerialization.jsonObject(with: data, options: .mutableContainers) as? [String: Any] {
                print(jsonResponse)
                let animationID = jsonResponse["animation_id"]
                let llmPrompt = jsonResponse["question"]
                return (String(format: "%@", animationID as! CVarArg), llmPrompt as! String)
                // handle json response
            } else {
                print("data maybe corrupted or in wrong format")
                return ("error", "error")
            }
            
        } catch {
            print(error.localizedDescription)
            return ("error", "error")
        }
    }
    func getAnimation(for animationID: String) async -> Data? {
        let defaultURLString = "https://e025-128-61-3-180.ngrok-free.app/get_animation/"
        let urlString = defaultURLString + animationID
        let url = URL(string: urlString)!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            return data
            
            
            
        } catch {
            print(error.localizedDescription)
            return nil
        }
    }
    
}

struct GIFActionView: View {
    var actionID: String
    var question: String
    @State var imageData: Data?
    var body: some View {
        VStack {
            
            Text("Recommended motion")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.top, 30)
                .onAppear {
                    Task {
                        let imageData = await getAnimation(for: actionID)
                        DispatchQueue.main.async {
                            self.imageData = imageData
                        }
                    }
                }
            ZStack {
                if imageData != nil {
                    Giffy(imageData: imageData!)
                        .frame(maxWidth: .infinity)
                        .offset(y: -50)
                    
                }
                VStack {
                    Spacer()
                    ScrollView {
                        Text(LocalizedStringKey(question))
                            .font(.body)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            
                    }
                    .frame(maxWidth: .infinity, maxHeight: 300)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                }
            }
        }
        .frame(maxHeight: .infinity, alignment: .top)
    
    }
    func getAnimation(for animationID: String) async -> Data? {
        let defaultURLString = "https://e025-128-61-3-180.ngrok-free.app/get_animation/"
        let urlString = defaultURLString + animationID
        let url = URL(string: urlString)!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            return data
        } catch {
            print(error.localizedDescription)
            return nil
        }
    }
}

struct HistoryView: View {
    @Binding var historyIDs: [String]
    @Binding var historyQuestions: [String]
    @Binding var dates: [String]

    var body: some View {
        VStack {
            Text("History")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.top, 30)
            if historyIDs.isEmpty {
                Spacer()
                    .frame(height: 50)
                Text("No history yet")
                    .font(.subheadline)
            } else {
                List {
                    ForEach(0..<historyIDs.count, id: \.self) { index in
                        NavigationLink(destination: GIFActionView(actionID: historyIDs[index], question: historyQuestions[index])) {
                            Text(dates[index])
                        }
                    }
                    .onDelete(perform: onDelete)
                }
            }
        }
        .frame(maxHeight: .infinity, alignment: .top)
        
    }
    private func onDelete(at offsets: IndexSet) {
        historyIDs.remove(atOffsets: offsets)
        UserDefaults.standard.set(historyIDs, forKey: "historyIDs")
        historyQuestions.remove(atOffsets: offsets)
        UserDefaults.standard.set(historyQuestions, forKey: "historyQuestions")
        dates.remove(atOffsets: offsets)
        UserDefaults.standard.set(dates, forKey: "historyDates")
    }
}

struct Wave: Shape {
    // allow SwiftUI to animate the wave phase
    var animatableData: Double {
        get { phase }
        set { self.phase = newValue }
    }
    
    // how high our waves should be
    var strength: Double
    
    // how frequent our waves should be
    var frequency: Double
    
    // how much to offset our waves horizontally
    var phase: Double
    
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath()
        
        // calculate some important values up front
        let width = Double(rect.width)
        let height = Double(rect.height)
        let midWidth = width / 2
        let midHeight = height / 2
        let oneOverMidWidth = 1 / midWidth
        
        // split our total width up based on the frequency
        let wavelength = width / frequency
        
        // start at the left center
        path.move(to: CGPoint(x: 0, y: midHeight))
        
        // now count across individual horizontal points one by one
        for x in stride(from: 0, through: width, by: 1) {
            // find our current position relative to the wavelength
            let relativeX = x / wavelength
            
            // find how far we are from the horizontal center
            let distanceFromMidWidth = x - midWidth
            
            // bring that into the range of -1 to 1
            let normalDistance = oneOverMidWidth * distanceFromMidWidth
            
            let parabola = -(normalDistance * normalDistance) + 1
            
            // calculate the sine of that position, adding our phase offset
            let sine = sin(relativeX + phase)
            
            // multiply that sine by our strength to determine final offset, then move it down to the middle of our view
            let y = parabola * strength * sine + midHeight
            
            // add a line to here
            path.addLine(to: CGPoint(x: x, y: y))
        }
        
        return Path(path.cgPath)
    }
}

struct RecordButton: View {
    @Binding var isRecording: Bool
    let buttonColor: Color
    let borderColor: Color
    let animation: Animation
    let startAction: () -> Void
    let stopAction: () -> Void
    
    init(
        isRecording: Binding<Bool>,
        buttonColor: Color = .indigo,
        borderColor: Color = .white,
        animation: Animation = .easeInOut(duration: 0.25),
        startAction: @escaping () -> Void,
        stopAction: @escaping () -> Void
    ) {
        self._isRecording = isRecording
        self.buttonColor = buttonColor
        self.borderColor = borderColor
        self.animation = animation
        self.startAction = startAction
        self.stopAction = stopAction
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                let minDimension = min(geometry.size.width, geometry.size.height)
                
                Button {
                    if isRecording {
                        deactivate()
                    } else {
                        activate()
                    }
                } label: {
                    RecordButtonShape(isRecording: isRecording)
                        .fill(buttonColor)
                }
                Circle()
                    .strokeBorder(lineWidth: minDimension * 0.05)
                    .foregroundColor(borderColor)
            }
        }
    }
    
    private func activate() {
        startAction()
        withAnimation(animation) {
            isRecording.toggle()
        }
    }
    
    private func deactivate() {
        stopAction()
        withAnimation(animation) {
            isRecording.toggle()
        }
    }
}

struct RecordButtonShape: Shape {
    var shapeRadius: CGFloat
    var distanceFromCardinal: CGFloat
    var b: CGFloat
    var c: CGFloat
    
    init(isRecording: Bool) {
        self.shapeRadius = isRecording ? 1.0 : 0.0
        self.distanceFromCardinal = isRecording ? 1.0 : 0.0
        self.b = isRecording ? 0.90 : 0.553
        self.c = isRecording ? 1.00 : 0.999
    }
    
    var animatableData: AnimatablePair<Double, AnimatablePair<Double, AnimatablePair<Double, Double>>> {
        get {
            AnimatablePair(Double(shapeRadius),
                           AnimatablePair(Double(distanceFromCardinal),
                                          AnimatablePair(Double(b), Double(c))))
        }
        set {
            shapeRadius = Double(newValue.first)
            distanceFromCardinal = Double(newValue.second.first)
            b = Double(newValue.second.second.first)
            c = Double(newValue.second.second.second)
        }
    }
    
    func path(in rect: CGRect) -> Path {
        let minDimension = min(rect.maxX, rect.maxY)
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = (minDimension / 2 * 0.82) - (shapeRadius * minDimension * 0.22)
        let movementFactor = 0.65
        
        let rightTop = CGPoint(x: center.x + radius, y: center.y - radius * movementFactor * distanceFromCardinal)
        let rightBottom = CGPoint(x: center.x + radius, y: center.y + radius * movementFactor * distanceFromCardinal)
        
        let topRight = CGPoint(x: center.x + radius * movementFactor * distanceFromCardinal, y: center.y - radius)
        let topLeft = CGPoint(x: center.x - radius * movementFactor * distanceFromCardinal, y: center.y - radius)
        
        let leftTop = CGPoint(x: center.x - radius, y: center.y - radius * movementFactor * distanceFromCardinal)
        let leftBottom = CGPoint(x: center.x - radius, y: center.y + radius * movementFactor * distanceFromCardinal)
        
        let bottomRight = CGPoint(x: center.x + radius * movementFactor * distanceFromCardinal, y: center.y + radius)
        let bottomLeft = CGPoint(x: center.x - radius * movementFactor * distanceFromCardinal, y: center.y + radius)
        
        let topRightControl1 = CGPoint(x: center.x + radius * c, y: center.y - radius * b)
        let topRightControl2 = CGPoint(x: center.x + radius * b, y: center.y - radius * c)
        
        let topLeftControl1 = CGPoint(x: center.x - radius * b, y: center.y - radius * c)
        let topLeftControl2 = CGPoint(x: center.x - radius * c, y: center.y - radius * b)
        
        let bottomLeftControl1 = CGPoint(x: center.x - radius * c, y: center.y + radius * b)
        let bottomLeftControl2 = CGPoint(x: center.x - radius * b, y: center.y + radius * c)
        
        let bottomRightControl1 = CGPoint(x: center.x + radius * b, y: center.y + radius * c)
        let bottomRightControl2 = CGPoint(x: center.x + radius * c, y: center.y + radius * b)
        
        var path = Path()
        
        path.move(to: rightTop)
        path.addCurve(to: topRight, control1: topRightControl1, control2: topRightControl2)
        path.addLine(to: topLeft)
        path.addCurve(to: leftTop, control1: topLeftControl1, control2: topLeftControl2)
        path.addLine(to: leftBottom)
        path.addCurve(to: bottomLeft, control1: bottomLeftControl1, control2: bottomLeftControl2)
        path.addLine(to: bottomRight)
        path.addCurve(to: rightBottom, control1: bottomRightControl1, control2: bottomRightControl2)
        path.addLine(to: rightTop)
        
        return path
    }
}

#Preview {
    InputView()
}
