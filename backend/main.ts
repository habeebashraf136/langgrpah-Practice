import express from 'express';
import { config } from 'dotenv';
import { ChatOpenRouter } from "@langchain/openrouter";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
config();

const app = express();
app.use(express.json());
const PORT = process.env.PORT || 4000;


const gemmaModel = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
  model: "deepseek/deepseek-chat-v3.1",
});

// console.log("Gemma Model Initialized:", gemmaModel);

const QwenModel = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
  model: "openai/gpt-oss-120b:free",
});

// console.log("Qwen Model Initialized:", QwenModel);

const hermesModel = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "",
  model: "nvidia/nemotron-3-super-120b-a12b:free",
})

// console.log("Hermes Model Initialized:", hermesModel);

const judgeSchema = z.object({
  Solution_1: z.number().min(0).max(10),
  Solution_2: z.number().min(0).max(10),
  Solution_1_reasoning: z.string(),
  Solution_2_reasoning: z.string(),
})

type JudgeResult = z.infer<typeof judgeSchema>;

const StateAnnotation = Annotation.Root({
    problem: Annotation<string>,
    Solution_1: Annotation<string>,
    Solution_2: Annotation<string>,
    judgement: Annotation<JudgeResult>,
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


const judge_Node = async (state: typeof StateAnnotation.State) => {
  const {problem, Solution_1, Solution_2} = state;

  const structuredJudge = hermesModel.withStructuredOutput(judgeSchema);

  const result = await structuredJudge.invoke(` 
    Problem: ${problem},
    Solution_1: ${Solution_1},
    Solution_2: ${Solution_2},

    Evaluate both solutions. Give each a score out of 10 and explain your reasoning.`
  );

  // console.log("Judge Result:", result); // LOG THE JUDGE RESULT
  return { judgement: result };
}

const graph = new StateGraph(StateAnnotation)
  .addNode("solution", solutionNode)
  .addNode("judge", judge_Node)
  .addEdge(START, "solution")
  .addEdge("solution", "judge")
  .addEdge("judge", END)
  .compile();


app.post("/solve", async (req, res) => {
  try {
    const { problem } = req.body;
    const result = await graph.invoke({ problem });
    // console.log(result);
    return res.json(result);
  } catch (error) {
    console.error("ACTUAL ERROR:", error); // ADD THIS LINE
    res.status(500).json({ error: "something went wrong" });
  }
});





app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});