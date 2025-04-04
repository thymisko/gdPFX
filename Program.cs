using System;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Threading;
using System.Linq;
using System.IO;

namespace gdPFX
{
    class Program
    {
        #region WinAPI Imports
        [DllImport("user32.dll")]
        static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, int dwExtraInfo);

        [DllImport("user32.dll")]
        static extern bool GetWindowRect(IntPtr hWnd, ref RECT rect);

        [DllImport("user32.dll")]
        static extern bool GetClientRect(IntPtr hWnd, ref RECT rect);

        [DllImport("user32.dll")]
        static extern bool ClientToScreen(IntPtr hWnd, ref POINT point);

        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct POINT
        {
            public int X;
            public int Y;
        }
        #endregion

        #region Configuration
        const int SCAN_WIDTH = 600;
        const int SCAN_HEIGHT = 400;
        const int PLAYER_SCAN_WIDTH = 150;
        const int PLAYER_SCAN_HEIGHT = 150;
        
        const int JUMP_KEY = 0x20;
        const int SECONDARY_KEY = 0x26;
        
        const int INPUT_SIZE = 15;
        const int OUTPUT_SIZE = 4;
        const int HIDDEN_SIZE = 64;
        const int MEMORY_SIZE = 50000;
        const int BATCH_SIZE = 128;
        
        const float INITIAL_EPSILON = 0.99f;
        const float FINAL_EPSILON = 0.01f;
        const float GAMMA = 0.99f;
        const float LEARNING_RATE = 0.01f;
        const int EXPLORATION_STEPS = 100000;
        #endregion

        #region Game State
        static float lastPlayerY = -1;
        static float lastPlayerX = -1;
        static float lastVelocityY = 0;
        static float totalProgress = 0;
        static float lastProgress = 0;
        static int consecutiveFramesWithoutPlayer = 0;
        
        static int gameWindowX, gameWindowY, gameWindowWidth, gameWindowHeight;
        static int scanX, scanY;
        
        static NeuralNetwork? brain;
        static List<TrainingSample> memory = new List<TrainingSample>();
        static List<TrainingSample> manualTrainingSamples = new List<TrainingSample>();
        static Random rand = new Random();
        
        static int totalAttempts = 0;
        static int bestScore = 0;
        static float bestTime = 0;
        static Stopwatch attemptTimer = new Stopwatch();
        static Stopwatch frameTimer = new Stopwatch();
        static float averageFrameTime = 0;
        
        static bool debugEnabled = true;
        static bool visualizeEnabled = true;
        static bool manualControlMode = false;
        static string currentCommand = "";
        static int debugFrameCounter = 0;
        static int visualizationInterval = 50;
        
        enum GameMode
        {
            Cube, Ship, Ball, UFO, Wave, Robot, Spider, Swing
        }
        
        static GameMode currentGameMode = GameMode.Cube;
        static Dictionary<GameMode, float> modeExplorationRates = new Dictionary<GameMode, float>();
        #endregion

        #region Neural Network Implementation
        class TrainingSample
        {
            public float[]? State { get; set; }
            public int Action { get; set; }
            public float Reward { get; set; }
            public float[]? NextState { get; set; }
            public bool IsDone { get; set; }
        }

        class NeuralNetwork
        {
            public int inputSize;
            public int hiddenSize;
            public int outputSize;
            public float[,] inputWeights;
            public float[,] hiddenWeights1;
            public float[,] hiddenWeights2;
            public float[] hiddenBias1;
            public float[] hiddenBias2;
            public float[] outputBias;
            private Random rand;

            public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
            {
                this.inputSize = inputSize;
                this.hiddenSize = hiddenSize;
                this.outputSize = outputSize;
                
                inputWeights = new float[inputSize, hiddenSize];
                hiddenWeights1 = new float[hiddenSize, hiddenSize];
                hiddenWeights2 = new float[hiddenSize, hiddenSize];
                hiddenBias1 = new float[hiddenSize];
                hiddenBias2 = new float[hiddenSize];
                outputBias = new float[outputSize];
                
                rand = new Random();

                float inputScale = (float)Math.Sqrt(2.0 / inputSize);
                float hiddenScale = (float)Math.Sqrt(2.0 / hiddenSize);
                
                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        inputWeights[i, j] = (float)(rand.NextDouble() * 2 - 1) * inputScale;
                    }
                }
                
                for (int i = 0; i < hiddenSize; i++)
                {
                    hiddenBias1[i] = (float)(rand.NextDouble() * 2 - 1) * 0.1f;
                    hiddenBias2[i] = (float)(rand.NextDouble() * 2 - 1) * 0.1f;
                    
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        hiddenWeights1[i, j] = (float)(rand.NextDouble() * 2 - 1) * hiddenScale;
                        hiddenWeights2[i, j] = (float)(rand.NextDouble() * 2 - 1) * hiddenScale;
                    }
                }
                
                for (int i = 0; i < outputSize; i++)
                {
                    outputBias[i] = (float)(rand.NextDouble() * 2 - 1) * 0.1f;
                }
            }

            public float[] Predict(float[] inputs)
            {
                float[] hidden1 = new float[hiddenSize];
                float[] hidden2 = new float[hiddenSize];
                float[] outputs = new float[outputSize];
                
                for (int i = 0; i < hiddenSize; i++)
                {
                    hidden1[i] = hiddenBias1[i];
                    for (int j = 0; j < inputSize; j++)
                    {
                        hidden1[i] += inputs[j] * inputWeights[j, i];
                    }
                    hidden1[i] = Math.Max(0, hidden1[i]);
                }
                
                for (int i = 0; i < hiddenSize; i++)
                {
                    hidden2[i] = hiddenBias2[i];
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        hidden2[i] += hidden1[j] * hiddenWeights1[j, i];
                    }
                    hidden2[i] = Math.Max(0, hidden2[i]);
                }
                
                for (int i = 0; i < outputSize; i++)
                {
                    outputs[i] = outputBias[i];
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        outputs[i] += hidden2[j] * hiddenWeights2[j, i];
                    }
                }
                
                return outputs;
            }

            public void Train(List<TrainingSample> batch, float learningRate = LEARNING_RATE, float gamma = GAMMA)
            {
                if (batch.Count == 0) return;
                
                float[,] inputWeightsGrad = new float[inputSize, hiddenSize];
                float[,] hiddenWeights1Grad = new float[hiddenSize, hiddenSize];
                float[,] hiddenWeights2Grad = new float[hiddenSize, hiddenSize];
                float[] hiddenBias1Grad = new float[hiddenSize];
                float[] hiddenBias2Grad = new float[hiddenSize];
                float[] outputBiasGrad = new float[outputSize];
                
                foreach (var sample in batch)
                {
                    if (sample.State == null || sample.NextState == null) continue;
                    
                    float[] hidden1 = new float[hiddenSize];
                    float[] hidden1PreActivation = new float[hiddenSize];
                    float[] hidden2 = new float[hiddenSize];
                    float[] hidden2PreActivation = new float[hiddenSize];
                    float[] outputs = new float[outputSize];
                    
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        hidden1PreActivation[i] = hiddenBias1[i];
                        for (int j = 0; j < inputSize; j++)
                        {
                            hidden1PreActivation[i] += sample.State[j] * inputWeights[j, i];
                        }
                        hidden1[i] = Math.Max(0, hidden1PreActivation[i]);
                    }
                    
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        hidden2PreActivation[i] = hiddenBias2[i];
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            hidden2PreActivation[i] += hidden1[j] * hiddenWeights1[j, i];
                        }
                        hidden2[i] = Math.Max(0, hidden2PreActivation[i]);
                    }
                    
                    for (int i = 0; i < outputSize; i++)
                    {
                        outputs[i] = outputBias[i];
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            outputs[i] += hidden2[j] * hiddenWeights2[j, i];
                        }
                    }
                    
                    float[] targetQ = new float[outputSize];
                    Array.Copy(outputs, targetQ, outputSize);
                    
                    float maxNextQ = 0;
                    if (!sample.IsDone)
                    {
                        float[] nextOutputs = Predict(sample.NextState);
                        maxNextQ = nextOutputs.Max();
                    }
                    
                    targetQ[sample.Action] = sample.Reward + (sample.IsDone ? 0 : gamma * maxNextQ);
                    
                    float[] outputDelta = new float[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        outputDelta[i] = outputs[i] - targetQ[i];
                        outputBiasGrad[i] += outputDelta[i];
                        
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            hiddenWeights2Grad[j, i] += hidden2[j] * outputDelta[i];
                        }
                    }
                    
                    float[] hidden2Delta = new float[hiddenSize];
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        if (hidden2PreActivation[i] > 0)
                        {
                            for (int j = 0; j < outputSize; j++)
                            {
                                hidden2Delta[i] += outputDelta[j] * hiddenWeights2[i, j];
                            }
                            
                            hiddenBias2Grad[i] += hidden2Delta[i];
                            
                            for (int j = 0; j < hiddenSize; j++)
                            {
                                hiddenWeights1Grad[j, i] += hidden1[j] * hidden2Delta[i];
                            }
                        }
                    }
                    
                    float[] hidden1Delta = new float[hiddenSize];
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        if (hidden1PreActivation[i] > 0)
                        {
                            for (int j = 0; j < hiddenSize; j++)
                            {
                                hidden1Delta[i] += hidden2Delta[j] * hiddenWeights1[i, j];
                            }
                            
                            hiddenBias1Grad[i] += hidden1Delta[i];
                            
                            for (int j = 0; j < inputSize; j++)
                            {
                                inputWeightsGrad[j, i] += sample.State[j] * hidden1Delta[i];
                            }
                        }
                    }
                }
                
                float batchLearningRate = learningRate / batch.Count;
                
                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        inputWeights[i, j] -= inputWeightsGrad[i, j] * batchLearningRate;
                    }
                }
                
                for (int i = 0; i < hiddenSize; i++)
                {
                    hiddenBias1[i] -= hiddenBias1Grad[i] * batchLearningRate;
                    hiddenBias2[i] -= hiddenBias2Grad[i] * batchLearningRate;
                    
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        hiddenWeights1[i, j] -= hiddenWeights1Grad[i, j] * batchLearningRate;
                        hiddenWeights2[i, j] -= hiddenWeights2Grad[i, j] * batchLearningRate;
                    }
                }
                
                for (int i = 0; i < outputSize; i++)
                {
                    outputBias[i] -= outputBiasGrad[i] * batchLearningRate;
                }
            }
        }
        #endregion

        #region Main Program
        static void Main(string[] args)
        {
            Console.Title = "gdPFX AI -- made with love by misko";
            Console.WriteLine("=== gdPFX ===");
            brain = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            
            foreach (GameMode mode in Enum.GetValues(typeof(GameMode)))
            {
                modeExplorationRates[mode] = INITIAL_EPSILON;
            }
            
            Process[] processes = Process.GetProcessesByName("GeometryDash");
            if (processes.Length == 0)
            {
                Console.WriteLine("Geometry Dash is not running! Please start the game first.");
                Console.ReadKey();
                return;
            }

            var gameWindow = processes[0];
            SetForegroundWindow(gameWindow.MainWindowHandle);
            
            RECT clientRect = new RECT();
            GetClientRect(gameWindow.MainWindowHandle, ref clientRect);
            
            POINT topLeft = new POINT();
            ClientToScreen(gameWindow.MainWindowHandle, ref topLeft);
            
            gameWindowX = topLeft.X;
            gameWindowY = topLeft.Y;
            gameWindowWidth = clientRect.Right;
            gameWindowHeight = clientRect.Bottom;
            
            scanX = gameWindowX + (gameWindowWidth / 2) - (SCAN_WIDTH / 2);
            scanY = gameWindowY + (gameWindowHeight / 2) - (SCAN_HEIGHT / 2);
            
            if (visualizeEnabled && !Directory.Exists("debug"))
            {
                Directory.CreateDirectory("debug");
            }

            Console.WriteLine($"Game window detected at: {gameWindowX}, {gameWindowY}, {gameWindowWidth}x{gameWindowHeight}");
            Console.WriteLine("Press any key to start training...");
            Console.ReadKey();

            WaitForLevelStart();
            RunTrainingLoop();
        }
        #endregion

        #region Core Game Logic
        static void WaitForLevelStart()
        {
            Console.WriteLine("Waiting for level start...");
            while (true)
            {
                using (var bmp = new Bitmap(100, 100))
                using (var g = Graphics.FromImage(bmp))
                {
                    g.CopyFromScreen(scanX + (SCAN_WIDTH/2) - 50, scanY + (SCAN_HEIGHT/2) - 50, 0, 0, bmp.Size);
                    
                    if (IsLevelRunning(bmp)) return;
                    
                    if (IsPlayButtonVisible(bmp))
                    {
                        Console.WriteLine("Detected play button! Starting level...");
                        Jump();
                        Thread.Sleep(1000);
                        return;
                    }
                }
                
                HandleUserInput();
                Thread.Sleep(100);
            }
        }

        static void RunTrainingLoop()
        {
            Console.WriteLine("Training Loop Started.");
            attemptTimer.Start();
            
            while (true)
            {
                totalAttempts++;
                int currentScore = 0;
                bool isAlive = true;
                float[]? lastState = null;
                int lastAction = 0;
                lastProgress = 0;
                consecutiveFramesWithoutPlayer = 0;
                
                attemptTimer.Restart();
                frameTimer.Restart();
                
                while (isAlive)
                {
                    frameTimer.Restart();
                    HandleUserInput();
                    
                    var state = GetGameState();
                    DetectGameMode(state);
                    
                    int action;
                    if (manualControlMode)
                    {
                        action = lastAction;
                        Thread.Sleep(16);
                    }
                    else
                    {
                        action = ChooseAction(state);
                        PerformGameModeSpecificAction(action);
                    }
                    
                    var nextState = GetGameState();
                    
                    if (manualControlMode)
                    {
                        RecordManualTrainingSample(state, action, nextState);
                    }
                    
                    float reward = CalculateReward(state, action, nextState);
                    bool isDone = DetectCrash(nextState) || DetectLevelComplete();
                    
                    if (lastState != null)
                    {
                        memory.Add(new TrainingSample {
                            State = lastState,
                            Action = lastAction,
                            Reward = reward,
                            NextState = nextState,
                            IsDone = isDone
                        });
                        
                        if (memory.Count > MEMORY_SIZE)
                        {
                            memory.RemoveAt(0);
                        }
                    }
                    
                    if (memory.Count + manualTrainingSamples.Count >= BATCH_SIZE && totalAttempts % 5 == 0)
                    {
                        var combinedBatch = new List<TrainingSample>();
                        combinedBatch.AddRange(memory);
                        combinedBatch.AddRange(manualTrainingSamples);
                        
                        if (combinedBatch.Count > BATCH_SIZE)
                        {
                            combinedBatch = combinedBatch
                                .OrderBy(x => rand.Next())
                                .Take(BATCH_SIZE)
                                .ToList();
                        }
                        
                        TrainFromMemory(combinedBatch);
                        manualTrainingSamples.Clear();
                    }
                    
                    currentScore = (int)(totalProgress * 100);
                    UpdateDisplay(currentScore);
                    
                    if (visualizeEnabled && debugFrameCounter % visualizationInterval == 0)
                    {
                        VisualizeAIView(state, action);
                    }
                    
                    lastState = nextState;
                    lastAction = action;
                    isAlive = !isDone;
                    
                    frameTimer.Stop();
                    float frameTime = frameTimer.ElapsedMilliseconds;
                    averageFrameTime = (averageFrameTime * 0.9f) + (frameTime * 0.1f);
                    debugFrameCounter++;
                    
                    int delay = Math.Max(0, 16 - (int)frameTime);
                    Thread.Sleep(delay);
                }
                
                attemptTimer.Stop();
                float attemptTime = attemptTimer.ElapsedMilliseconds / 1000.0f;
                if (attemptTime > bestTime)
                {
                    bestTime = attemptTime;
                }
                
                modeExplorationRates[currentGameMode] = Math.Max(FINAL_EPSILON, 
                    modeExplorationRates[currentGameMode] * (float)Math.Pow(FINAL_EPSILON / INITIAL_EPSILON, 1.0 / EXPLORATION_STEPS));
                
                Thread.Sleep(1000);
                Jump();
                Thread.Sleep(2000);
            }
        }
        #endregion

        #region User Interaction
        static void HandleUserInput()
        {
            while (Console.KeyAvailable)
            {
                var key = Console.ReadKey(true).Key;
                
                if (manualControlMode)
                {
                    switch (key)
                    {
                        case ConsoleKey.J:
                            currentCommand = "JUMP";
                            PerformGameModeSpecificAction(1);
                            break;
                        case ConsoleKey.H:
                            currentCommand = "HOLD";
                            PerformGameModeSpecificAction(2);
                            break;
                        case ConsoleKey.R:
                            currentCommand = "RELEASE";
                            PerformGameModeSpecificAction(3);
                            break;
                        case ConsoleKey.N:
                            currentCommand = "NONE";
                            PerformGameModeSpecificAction(0);
                            break;
                        case ConsoleKey.Escape:
                            manualControlMode = false;
                            Console.WriteLine("Exited manual control mode");
                            break;
                    }
                }
                else
                {
                    switch (key)
                    {
                        case ConsoleKey.D:
                            debugEnabled = !debugEnabled;
                            Console.WriteLine($"Debug mode: {(debugEnabled ? "ON" : "OFF")}");
                            break;
                        case ConsoleKey.V:
                            manualControlMode = true;
                            currentCommand = "";
                            Console.WriteLine("Entered manual control mode");
                            break;
                        case ConsoleKey.S:
                            SaveNetwork();
                            break;
                        case ConsoleKey.L:
                            if (LoadNetwork())
                                Console.WriteLine("Network loaded successfully!");
                            else
                                Console.WriteLine("No saved network found.");
                            break;
                        case ConsoleKey.R:
                            modeExplorationRates[currentGameMode] = 0.5f;
                            Console.WriteLine($"Reset e xploration rate for {currentGameMode}");
                            break;
                        case ConsoleKey.M:
                            modeExplorationRates[currentGameMode] = 0.10f;
                            Console.WriteLine($"Set exploration rate for {currentGameMode} to 0.10");
                            break;  
                        case ConsoleKey.Q:
                            Console.WriteLine("Quitting...");
                            Environment.Exit(0);
                            break;
                        case ConsoleKey.H:
                            Console.WriteLine("\n=== Commands ===");
                            Console.WriteLine("D - Toggle debug\nV - Manual mode\nS - Save\nL - Load\nR - Reset rate\nQ - Quit\nH - Help");
                            break;
                    }
                }
            }
        }

        static void UpdateDisplay(int score)
        {
            if (totalAttempts % 10 == 0 || manualControlMode)
            {
                Console.WriteLine($"=== gdPFX AI ===");
                Console.WriteLine($"Attempt: {totalAttempts} | Mode: {currentGameMode}");
                Console.WriteLine($"Best Score: {bestScore} | Current: {score}");
                Console.WriteLine($"Best Time: {bestTime:F1}s | Current: {attemptTimer.ElapsedMilliseconds/1000.0f:F1}s");
                Console.WriteLine($"Exploration: {modeExplorationRates[currentGameMode]:P0} | Memory: {memory.Count}/{MEMORY_SIZE}");
                Console.WriteLine($"Frame Time: {averageFrameTime:F1}ms | FPS: {1000.0f/Math.Max(1, averageFrameTime):F0}");
                Console.WriteLine("Progress: " + new string('#', score / 5));

                if (manualControlMode)
                {
                    Console.WriteLine("\n=== MANUAL CONTROL MODE ===");
                    Console.WriteLine("Commands: J=Jump, H=Hold, R=Release, N=None");
                    Console.WriteLine("Press ESC to exit manual mode");
                    Console.WriteLine($"Current command: {currentCommand}");
                }

                if (score > bestScore) bestScore = score;
            }
        }

        static void RecordManualTrainingSample(float[] state, int action, float[] nextState)
        {
            if (!manualControlMode) return;

            float reward = CalculateReward(state, action, nextState);
            bool isDone = DetectCrash(nextState) || DetectLevelComplete();

            manualTrainingSamples.Add(new TrainingSample {
                State = state,
                Action = action,
                Reward = reward,
                NextState = nextState,
                IsDone = isDone
            });

            if (debugEnabled)
            {
                Console.WriteLine($"Recorded manual action: {action} with reward: {reward:F2}");
            }
        }
        #endregion

        #region Game Mechanics
        static void PerformGameModeSpecificAction(int action)
        {
            const uint KEYEVENTF_KEYUP = 0x0002;
            
            switch (currentGameMode)
            {
                case GameMode.Ship:
                case GameMode.UFO:
                case GameMode.Wave:
                    switch (action)
                    {
                        case 0:
                            keybd_event((byte)SECONDARY_KEY, 0, KEYEVENTF_KEYUP, 0);
                            break;
                        case 1:
                        case 2:
                            keybd_event((byte)SECONDARY_KEY, 0, 0, 0);
                            break;
                        case 3:
                            keybd_event((byte)SECONDARY_KEY, 0, KEYEVENTF_KEYUP, 0);
                            break;
                    }
                    break;
                    
                case GameMode.Ball:
                case GameMode.Spider:
                    if (action == 1)
                    {
                        keybd_event((byte)JUMP_KEY, 0, 0, 0);
                        Thread.Sleep(30);
                        keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                    }
                    break;
                    
                case GameMode.Robot:
                    if (action == 1)
                    {
                        keybd_event((byte)JUMP_KEY, 0, 0, 0);
                        Thread.Sleep(50);
                        keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                    }
                    else if (action == 2)
                    {
                        keybd_event((byte)JUMP_KEY, 0, 0, 0);
                        Thread.Sleep(200);
                        keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                    }
                    break;
                    
                case GameMode.Swing:
                    if (action == 1)
                    {
                        keybd_event((byte)JUMP_KEY, 0, 0, 0);
                        Thread.Sleep(30);
                        keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                    }
                    break;
                    
                default:
                    switch (action)
                    {
                        case 0:
                            keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                            break;
                        case 1:
                            keybd_event((byte)JUMP_KEY, 0, 0, 0);
                            Thread.Sleep(50);
                            keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                            break;
                        case 3:
                            keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
                            break;
                    }
                    break;
            }
        }

        static void Jump()
        {
            const uint KEYEVENTF_KEYUP = 0x0002;
            keybd_event((byte)JUMP_KEY, 0, 0, 0);
            Thread.Sleep(50);
            keybd_event((byte)JUMP_KEY, 0, KEYEVENTF_KEYUP, 0);
        }
        #endregion

        #region Game State Detection
        static bool IsLevelRunning(Bitmap bmp)
{
    int playerPixels = 0;
    
    for (int x = 0; x < bmp.Width; x++)
    {
        for (int y = 0; y < bmp.Height; y++)
        {
            if (IsPlayerPixel(bmp, x, y))
            {
                playerPixels++;
                if (playerPixels > 10) return true;
            }
        }
    }
    
    return false;
}

        static bool IsPlayButtonVisible(Bitmap bmp)
{
    int greenPixels = 0;
    
    for (int x = 0; x < bmp.Width; x++)
    {
        for (int y = 0; y < bmp.Height; y++)
        {
            Color pixel = bmp.GetPixel(x, y);
            if (pixel.G > 200 && pixel.R < 100 && pixel.B < 100)
            {
                greenPixels++;
                if (greenPixels > 50) return true;
            }
        }
    }
    
    return false;
}
        #endregion

#region Training and Evaluation
static int ChooseAction(float[] state)
{
    if (brain == null) return 0;

    if (rand.NextDouble() < modeExplorationRates[currentGameMode])
    {
        return rand.Next(OUTPUT_SIZE);
    }

    float[] qValues = brain.Predict(state);
    int bestAction = 0;
    float bestValue = qValues[0];
    
    for (int i = 1; i < qValues.Length; i++)
    {
        if (qValues[i] > bestValue)
        {
            bestValue = qValues[i];
            bestAction = i;
        }
    }
    
    return bestAction;
}
#endregion

        static float[] GetGameState()
        {
            using (var bmp = new Bitmap(SCAN_WIDTH, SCAN_HEIGHT))
            using (var g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(scanX, scanY, 0, 0, bmp.Size);
                
                float playerX = GetPlayerX(bmp);
                float playerY = GetPlayerY(bmp);
                float playerVelocityY = GetPlayerVelocityY(bmp);
                
                List<float[]> obstacles = GetObstacles(bmp, playerX, playerY);
                
                float[] state = new float[INPUT_SIZE];
                
                state[0] = playerX / SCAN_WIDTH;
                state[1] = playerY / SCAN_HEIGHT;
                state[2] = playerVelocityY / 10.0f;
                state[3] = (float)currentGameMode / 8.0f;
                
                for (int i = 0; i < Math.Min(obstacles.Count, 3); i++)
                {
                    state[4 + i*3] = obstacles[i][0] / SCAN_WIDTH;
                    state[5 + i*3] = obstacles[i][1] / SCAN_HEIGHT;
                    state[6 + i*3] = obstacles[i][2];
                }
                
                for (int i = obstacles.Count; i < 3; i++)
                {
                    state[4 + i*3] = 1.0f;
                    state[5 + i*3] = 0.5f;
                    state[6 + i*3] = 0.1f;
                }
                
                state[13] = (playerY < SCAN_HEIGHT / 3) ? 1.0f : 0.0f;
                state[14] = (playerY > 2 * SCAN_HEIGHT / 3) ? 1.0f : 0.0f;
                
                return state;
            }
        }

        static void DetectGameMode(float[] state)
        {
            using (var bmp = new Bitmap(PLAYER_SCAN_WIDTH, PLAYER_SCAN_HEIGHT))
            using (var g = Graphics.FromImage(bmp))
            {
                float playerX = state[0] * SCAN_WIDTH;
                float playerY = state[1] * SCAN_HEIGHT;
                
                int captureX = (int)playerX - (PLAYER_SCAN_WIDTH / 2);
                int captureY = (int)playerY - (PLAYER_SCAN_HEIGHT / 2);
                
                captureX = Math.Max(0, Math.Min(SCAN_WIDTH - PLAYER_SCAN_WIDTH, captureX));
                captureY = Math.Max(0, Math.Min(SCAN_HEIGHT - PLAYER_SCAN_HEIGHT, captureY));
                
                g.CopyFromScreen(scanX + captureX, scanY + captureY, 0, 0, bmp.Size);
                
                int bluePixels = 0;
                int yellowPixels = 0;
                int purplePixels = 0;
                int redPixels = 0;
                int blackPixels = 0;
                
                for (int x = 0; x < bmp.Width; x++)
                {
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        Color pixel = bmp.GetPixel(x, y);
                        
                        if (pixel.B > 180 && pixel.R < 120 && pixel.G < 120) bluePixels++;
                        if (pixel.R > 180 && pixel.G > 180 && pixel.B < 120) yellowPixels++;
                        if (pixel.R > 150 && pixel.B > 180 && pixel.G < 120) purplePixels++;
                        if (pixel.R > 180 && pixel.G < 120 && pixel.B < 120) redPixels++;
                        if (pixel.GetBrightness() < 0.2f) blackPixels++;
                    }
                }
                
                if (bluePixels > 100 && blackPixels < 50)
                    currentGameMode = GameMode.Ship;
                else if (yellowPixels > 100)
                    currentGameMode = GameMode.Ball;
                else if (purplePixels > 100)
                    currentGameMode = GameMode.Wave;
                else if (redPixels > 100)
                    currentGameMode = GameMode.UFO;
                else
                    currentGameMode = GameMode.Cube;
            }
        }

        static List<float[]> GetObstacles(Bitmap bmp, float playerX, float playerY)
        {
            List<float[]> obstacles = new List<float[]>();
            int playerXInt = (int)playerX;
            
            for (int x = playerXInt + 20; x < SCAN_WIDTH; x += 5)
            {
                for (int y = 0; y < SCAN_HEIGHT; y += 5)
                {
                    if (IsObstacle(bmp, x, y))
                    {
                        int obstacleWidth = 1;
                        while (x + obstacleWidth < SCAN_WIDTH && IsObstacle(bmp, x + obstacleWidth, y))
                        {
                            obstacleWidth++;
                        }
                        
                        obstacles.Add(new float[] { x - playerXInt, y, obstacleWidth });
                        x += obstacleWidth;
                        break;
                    }
                }
                
                if (obstacles.Count >= 5) break;
            }
            
            obstacles.Sort((a, b) => a[0].CompareTo(b[0]));
            return obstacles;
        }

        static bool IsObstacle(Bitmap bmp, int x, int y)
        {
            if (x < 0 || y < 0 || x >= bmp.Width || y >= bmp.Height)
                return false;
                
            Color pixel = bmp.GetPixel(x, y);
            float brightness = pixel.GetBrightness();
            float saturation = GetSaturation(pixel);
            return brightness < 0.4f && saturation > 0.3f;
        }

        static float GetPlayerX(Bitmap bmp)
        {
            for (int x = 50; x < SCAN_WIDTH / 3; x++)
            {
                for (int y = 0; y < SCAN_HEIGHT; y++)
                {
                    if (IsPlayerPixel(bmp, x, y))
                    {
                        return x;
                    }
                }
            }
            return SCAN_WIDTH / 4;
        }

        static float GetPlayerY(Bitmap bmp)
        {
            int playerX = (int)GetPlayerX(bmp);
            int minY = SCAN_HEIGHT;
            int maxY = 0;
            int playerPixels = 0;
            
            for (int x = playerX - 10; x < playerX + 10; x++)
            {
                if (x < 0 || x >= SCAN_WIDTH) continue;
                
                for (int y = 0; y < SCAN_HEIGHT; y++)
                {
                    if (IsPlayerPixel(bmp, x, y))
                    {
                        minY = Math.Min(minY, y);
                        maxY = Math.Max(maxY, y);
                        playerPixels++;
                    }
                }
            }
            
            if (playerPixels > 5)
            {
                consecutiveFramesWithoutPlayer = 0;
                return (minY + maxY) / 2.0f;
            }
            else
            {
                consecutiveFramesWithoutPlayer++;
                return lastPlayerY >= 0 ? lastPlayerY : SCAN_HEIGHT / 2.0f;
            }
        }

        static bool IsPlayerPixel(Bitmap bmp, int x, int y)
        {
            if (x < 0 || y < 0 || x >= bmp.Width || y >= bmp.Height)
                return false;
                
            Color pixel = bmp.GetPixel(x, y);
            float brightness = pixel.GetBrightness();
            float saturation = GetSaturation(pixel);
            return brightness > 0.7f && saturation > 0.5f;
        }

        static float GetPlayerVelocityY(Bitmap bmp)
        {
            float currentY = GetPlayerY(bmp);
    
            if (lastPlayerY < 0)
            {
                lastPlayerY = currentY;
                return 0;
            }

            float velocity = (currentY - lastPlayerY) / 0.016f;
            lastVelocityY = velocity;
            return velocity;
        }

        static float GetSaturation(Color color)
        {
            float r = color.R / 255.0f;
            float g = color.G / 255.0f;
            float b = color.B / 255.0f;
            
            float max = Math.Max(r, Math.Max(g, b));
            float min = Math.Min(r, Math.Min(g, b));
            
            if (max == 0) return 0;
            return (max - min) / max;
        }

        #region Training and Evaluation
        static void TrainFromMemory(List<TrainingSample> batch)
        {
            if (batch.Count < BATCH_SIZE || brain == null) return;
            
            float adaptiveLearningRate = LEARNING_RATE * (1.0f - (totalAttempts / (float)EXPLORATION_STEPS));
            brain.Train(batch, adaptiveLearningRate);
            
            if (debugEnabled && totalAttempts % 50 == 0)
            {
                Console.WriteLine($"Trained with batch of {batch.Count}. Learning rate: {adaptiveLearningRate:F5}");
            }
        }

        static float CalculateReward(float[] state, int action, float[] nextState)
        {
            float reward = 0.01f;
            
            float progress = CalculateProgress(nextState);
            if (progress > lastProgress)
            {
                reward += (progress - lastProgress) * 10.0f;
                lastProgress = progress;
            }
            
            if (DetectCrash(nextState))
            {
                reward -= 1.0f;
            }
            
            switch (currentGameMode)
            {
                case GameMode.Ship:
                case GameMode.Wave:
                    float yPos = nextState[1];
                    reward += 0.02f * (1.0f - Math.Abs(yPos - 0.5f) * 2.0f);
                    break;
                    
                case GameMode.Ball:
                case GameMode.Spider:
                    if (nextState[13] > 0.5f || nextState[14] > 0.5f)
                    {
                        reward += 0.02f;
                    }
                    break;
            }
            
            if (action != 0 && currentGameMode == GameMode.Cube)
            {
                reward -= 0.005f;
            }
            
            return reward;
        }

        static bool DetectCrash(float[] state)
        {
            if (consecutiveFramesWithoutPlayer > 5)
            {
                return true;
            }
            
            using (var bmp = new Bitmap(100, 100))
            using (var g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(
                    scanX + (SCAN_WIDTH / 2) - 50,
                    scanY + (SCAN_HEIGHT / 2) - 50,
                    0, 0, bmp.Size);
                
                int redPixels = 0;
                int whitePixels = 0;
                
                for (int x = 0; x < bmp.Width; x++)
                {
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        Color pixel = bmp.GetPixel(x, y);
                        
                        if (pixel.R > 200 && pixel.G < 100 && pixel.B < 100)
                        {
                            redPixels++;
                        }
                        
                        if (pixel.R > 220 && pixel.G > 220 && pixel.B > 220)
                        {
                            whitePixels++;
                        }
                    }
                }
                
                return (redPixels > 100 || whitePixels > 500);
            }
        }

        static bool DetectLevelComplete()
        {
            using (var bmp = new Bitmap(150, 50))
            using (var g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(
                    scanX + SCAN_WIDTH - 150,
                    scanY + (SCAN_HEIGHT / 2) - 25,
                    0, 0, bmp.Size);
                
                int bluePixels = 0;
                int lightPixels = 0;
                
                for (int x = 0; x < bmp.Width; x++)
                {
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        Color pixel = bmp.GetPixel(x, y);
                        
                        if (pixel.B > 180 && pixel.R < 100 && pixel.G < 150)
                        {
                            bluePixels++;
                        }
                        
                        if (pixel.GetBrightness() > 0.9f)
                        {
                            lightPixels++;
                        }
                    }
                }
                
                return (bluePixels > 200 || lightPixels > 300);
            }
        }

        static float CalculateProgress(float[] state)
        {
            float playerX = state[0] * SCAN_WIDTH;
            
            if (lastPlayerX < 0)
            {
                lastPlayerX = playerX;
                return totalProgress;
            }
            
            if (playerX > lastPlayerX)
            {
                totalProgress += (playerX - lastPlayerX) / 5000.0f;
            }
            
            lastPlayerX = playerX;
            return totalProgress;
        }
        #endregion

        #region Visualization and IO
        static void VisualizeAIView(float[] state, int action)
        {
            using (var bmp = new Bitmap(SCAN_WIDTH, SCAN_HEIGHT))
            using (var g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(scanX, scanY, 0, 0, bmp.Size);
                
                float playerX = state[0] * SCAN_WIDTH;
                float playerY = state[1] * SCAN_HEIGHT;
                g.FillEllipse(Brushes.Red, playerX - 5, playerY - 5, 10, 10);
                
                for (int i = 0; i < 3; i++)
                {
                    float obstacleX = playerX + (state[4 + i*3] * SCAN_WIDTH);
                    float obstacleY = state[5 + i*3] * SCAN_HEIGHT;
                    float obstacleWidth = state[6 + i*3] * SCAN_WIDTH;
                    
                    if (obstacleX < SCAN_WIDTH * 0.9f)
                    {
                        g.DrawRectangle(Pens.Yellow, obstacleX, obstacleY - 10, obstacleWidth, 20);
                    }
                }
                
                string actionText = action switch
                {
                    0 => "NONE",
                    1 => "JUMP",
                    2 => "HOLD",
                    3 => "RELEASE",
                    _ => "UNKNOWN"
                };
                
                g.DrawString(actionText, new Font("Arial", 12), Brushes.White, 10, 10);
                g.DrawString(currentGameMode.ToString(), new Font("Arial", 12), Brushes.White, 10, 30);
                
                try
                {
                    bmp.Save($"debug/frame_{totalAttempts}_{debugFrameCounter}.png");
                }
                catch (Exception ex)
                {
                    if (debugEnabled) Console.WriteLine($"Error saving debug image: {ex.Message}");
                }
            }
        }

        static void SaveNetwork(string filename = "gdai_network.dat")
        {
            try
            {
                using (var stream = File.Create(filename))
                using (var writer = new BinaryWriter(stream))
                {
                    if (brain == null) return;
                    
                    writer.Write(brain.inputSize);
                    writer.Write(brain.hiddenSize);
                    writer.Write(brain.outputSize);
                    
                    for (int i = 0; i < brain.inputSize; i++)
                    {
                        for (int j = 0; j < brain.hiddenSize; j++)
                        {
                            writer.Write(brain.inputWeights[i, j]);
                        }
                    }
                    
                    for (int i = 0; i < brain.hiddenSize; i++)
                    {
                        writer.Write(brain.hiddenBias1[i]);
                        writer.Write(brain.hiddenBias2[i]);
                        
                        for (int j = 0; j < brain.hiddenSize; j++)
                        {
                            writer.Write(brain.hiddenWeights1[i, j]);
                            writer.Write(brain.hiddenWeights2[i, j]);
                        }
                    }
                    
                    for (int i = 0; i < brain.outputSize; i++)
                    {
                        writer.Write(brain.outputBias[i]);
                    }
                    
                    Console.WriteLine($"Network saved to {filename}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving network: {ex.Message}");
            }
        }

        static bool LoadNetwork(string filename = "gdai_network.dat")
        {
            if (!File.Exists(filename)) return false;
            
            try
            {
                using (var stream = File.OpenRead(filename))
                using (var reader = new BinaryReader(stream))
                {
                    if (brain == null) return false;
                    
                    int inputSize = reader.ReadInt32();
                    int hiddenSize = reader.ReadInt32();
                    int outputSize = reader.ReadInt32();
                    
                    if (inputSize != brain.inputSize || hiddenSize != brain.hiddenSize || outputSize != brain.outputSize)
                    {
                        Console.WriteLine("Network architecture mismatch!");
                        return false;
                    }
                    
                    for (int i = 0; i < brain.inputSize; i++)
                    {
                        for (int j = 0; j < brain.hiddenSize; j++)
                        {
                            brain.inputWeights[i, j] = reader.ReadSingle();
                        }
                    }
                    
                    for (int i = 0; i < brain.hiddenSize; i++)
                    {
                        brain.hiddenBias1[i] = reader.ReadSingle();
                        brain.hiddenBias2[i] = reader.ReadSingle();
                        
                        for (int j = 0; j < brain.hiddenSize; j++)
                        {
                            brain.hiddenWeights1[i, j] = reader.ReadSingle();
                            brain.hiddenWeights2[i, j] = reader.ReadSingle();
                        }
                    }
                    
                    for (int i = 0; i < brain.outputSize; i++)
                    {
                        brain.outputBias[i] = reader.ReadSingle();
                    }
                    
                    return true;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading network: {ex.Message}");
                return false;
            }
        }
        #endregion
    }
}