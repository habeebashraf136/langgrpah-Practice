import express from 'express';
import { config } from 'dotenv';
import { ChatOpenRouter } from "@langchain/openrouter";
import { HumanMessage } from "@langchain/core/messages";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
config();

const app = express();
app.use(express.json());
const PORT = process.env.PORT || 4000;


const gemmaModel = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
  model: "deepseek/deepseek-chat-v3.1",
});

const QwenModel = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
  model: "openai/gpt-oss-120b:free",
});

const StateAnnotation = Annotation.Root({
    problem: Annotation<string>({
        reducer: (_, b) => b,
        default: () => "",
    }),
    Solution_1: Annotation<string>({
        reducer: (_, b) => b,
        default: () => "",
    }),
    Solution_2: Annotation<string>({
        reducer: (_, b) => b,
        default: () => "",
    }),
})




const solutionNode = async (state: typeof StateAnnotation.State) => {
    const [QwenResponse,gemmaResponse] = await Promise.all([
        QwenModel.invoke([new HumanMessage(state.problem)]),
        gemmaModel.invoke([new HumanMessage(state.problem)]),
    ])

    return {
        Solution_1: QwenResponse.content as string,
        Solution_2: gemmaResponse.content as string,
    }
}

const graph = new StateGraph(StateAnnotation)
  .addNode("solution", solutionNode)
  .addEdge(START, "solution")
  .addEdge("solution", END)
  .compile();



app.post("/solve", async (req, res) => {
  try {
    const { problem } = req.body;
    const result = await graph.invoke({ problem });
    console.log(result);
    return res.json(result);
  } catch (error) {
    console.error("ACTUAL ERROR:", error); // ADD THIS LINE
    res.status(500).json({ error: "something went wrong" });
  }
});





app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});