# PhysioFix: Interactive GenAI Physiotherapy Movement Trainer

PhysioFix is your AI physiotherapy trainer, using GenAI text2motion models with clinician-approved data for personalized therapy exercises and rehabilitation plans for all your physiotherapy problems.

## Inspiration
Recovering from physical injuries or managing chronic pain often requires expert guidance, but access to physiotherapy can be limited due to cost, time, or location. As students who are passionate about sports and exercise, we all have personal experiences with the lack of accessible physiotherapy care. Thus, we wanted to address this gap in healthcare and aimed to create a smart, AI-driven physiotherapy assistant that provides personalized rehabilitation support anytime, anywhere. By combining GenAI models and motion synthesis, we aimed to make physiotherapy more accessible, interactive, and engaging for users seeking effective recovery solutions.

## What it does
Our app takes voice input from users describing their pain or mobility issues. It processes it through a pipeline of Gen AI models and Retrieval-Augmented Generation (RAG) GPT frameworks and retrieves relevant physiotherapy insights from licensed physiotherapy databases, ensuring that the analysis is precise and tailored to the user's needs. This analysis is then refined and used to generate a GIF demonstration of the recommended exercises. Users receive both a clear, AI-driven movement guide and a detailed physiotherapy analysis, helping them perform exercises correctly and effectively.

## How we built it
We developed the app using Swift, integrating speech-to-text functionality to capture user input using Apple's SFSpeechRecognizer and AVAudioEngine. The backend is powered by a flask REST API that connects to a LangChain script performing Retrieval Augmented Generation with a GPT 4o model and a Chroma vector database, ensuring that the physiotherapy analysis is context-aware and personalized. The analysis is then processed to be formatted for the text-to-motion model SATO (Stable Text-to-Motion), which generates a GIF that is displayed in the app alongside the text-based analysis. The result is an intuitive, AI-powered physiotherapy assistant that combines natural language understanding, retrieval-based AI, and motion synthesis for a smarter rehabilitation experience.

## Challenges we ran into
One major challenge was the lack of CUDA/NVIDIA GPUs, which made running highly intensive models difficult. Since Text-to-Motion models and large language models require significant computational power, we had to optimize performance and explore cloud-based solutions like AWS GPU EC2 instances to handle these workloads efficiently.

Another challenge was integrating multiple AI components into a seamless pipeline. Our system involves speech recognition, a RAG GPT model, a Chroma database, text formatting for motion synthesis, and GIF generation—all of which needed to work together smoothly, but had conflicting package and library requirements. Ensuring fast response times, proper data flow, and compatibility between these models required extensive debugging and architectural refinement.

Despite these hurdles, we successfully built a functional pipeline by leveraging cloud resources, optimizing API calls, and refining model outputs to create an interactive and responsive physiotherapy assistant.
## Accomplishments that we're proud of
We’re proud of successfully integrating multiple AI-driven technologies into a cohesive physiotherapy assistant. From speech recognition and RAG-based retrieval to Text-to-Motion GIF generation, we built a functional pipeline that delivers personalized, real-time physiotherapy guidance.

Another key achievement was overcoming hardware limitations by optimizing performance and leveraging cloud-based solutions. Despite not having access to CUDA/NVIDIA GPUs, we ensured smooth processing across our GPT models, Chroma database, and motion synthesis models.

Additionally, refining the user experience was a big win. We transformed complex AI processes into an intuitive app where users can describe pain, receive tailored analysis, and see an AI-generated motion demo—all within seconds.
## What we learned
We gained valuable experience in building AI-powered pipelines, particularly in combining language models, retrieval-augmented generation, and motion synthesis into a single workflow.

We also learned about the challenges of scaling AI applications without high-end GPUs, pushing us to explore cloud-based inference, model optimizations, and efficient API structuring.

Lastly, working across multiple technologies reinforced the importance of modular design—breaking down complex AI tasks into manageable components helped us debug issues faster and ensure seamless integration.

## What's next for PhysioFix: Interactive GenAI Physiotherapy Movement Trainer
We hope to deploy this application on the AppStore/Google Play Store and also expand accessibility as a website application, for a low price subscription price. Our end goal is to increase the accessibility of physiotherapy to everyone across the world, at a fraction of the cost!
