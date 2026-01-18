import {
  AgentKit,
  CdpEvmWalletProvider,
  CdpApiActionProvider,
  CdpEvmWalletActionProvider,
} from "@coinbase/agentkit";
import { getLangChainTools } from "@coinbase/agentkit-langchain";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { DynamicTool } from "@langchain/core/tools";
import { TwitterApi } from "twitter-api-v2";
import * as dotenv from "dotenv";
import * as fs from "fs";

dotenv.config();

const WALLET_DATA_FILE = "wallet_data.txt";

export class BaseAIAgent {
  private agentKit: AgentKit | undefined;
  private genAI: GoogleGenerativeAI | undefined;
  private model: any;
  private tools: Map<string, DynamicTool> = new Map();
  private twitterClient: TwitterApi;
  private processedTweets: Set<string> = new Set();
  private lastProcessedMentionId: string | null = null;

  constructor() {
    this.twitterClient = new TwitterApi({
      appKey: process.env.TWITTER_API_KEY!,
      appSecret: process.env.TWITTER_API_SECRET!,
      accessToken: process.env.TWITTER_ACCESS_TOKEN!,
      accessSecret: process.env.TWITTER_ACCESS_SECRET!,
    });
  }

  async initialize() {
    // Configure CDP Wallet Provider
    const walletProvider = await CdpEvmWalletProvider.configureWithWallet({
      apiKeyId: process.env.CDP_API_KEY_NAME!,
      apiKeySecret: process.env.CDP_API_KEY_PRIVATE_KEY?.replace(/\\n/g, "\n"),
      networkId: "base-sepolia",
    });

    // Initialize AgentKit with action providers
    this.agentKit = await AgentKit.from({
      walletProvider,
      actionProviders: [
        new CdpApiActionProvider(),
        new CdpEvmWalletActionProvider(),
      ],
    });

    // Initialize Gemini API directly
    if (!process.env.GOOGLE_API_KEY) {
      throw new Error("GOOGLE_API_KEY must be set in .env file");
    }

    this.genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    this.model = this.genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

    // Get LangChain tools from AgentKit
    const agentKitTools = await getLangChainTools(this.agentKit);
    for (const tool of agentKitTools) {
      this.tools.set(tool.name, tool as DynamicTool);
    }

    // Add custom Twitter tool
    const twitterTool = new DynamicTool({
      name: "send_tweet",
      description:
        "Send a tweet. Input format: for new tweet just send the text, for replies use format 'REPLY:tweetId:text'",
      func: async (input: string) => {
        try {
          if (input.startsWith("REPLY:")) {
            const [_, replyToId, text] = input.split(":", 3);
            const result = await this.twitterClient.v2.reply(text, replyToId);
            return `Tweet sent as reply: ${result.data.id}`;
          } else {
            const result = await this.twitterClient.v2.tweet(input);
            return `Tweet sent: ${result.data.id}`;
          }
        } catch (error) {
          console.error("Error sending tweet:", error);
          throw new Error("Failed to send tweet");
        }
      },
    });

    this.tools.set("send_tweet", twitterTool);

    // Export and save wallet data
    const walletData = await walletProvider.exportWallet();
    fs.writeFileSync(WALLET_DATA_FILE, JSON.stringify(walletData));

    console.log("Agent initialized on Base Sepolia");
  }

  private async callLLM(userMessage: string) {
    if (!this.model) throw new Error("LLM not initialized");
    const result = await this.model.generateContent(userMessage);
    const text = result.response.text();
    return text;
  }


  async handleTweet(
    tweetText: string,
    authorUsername: string,
    tweetId: string
  ) {
    if (this.processedTweets.has(tweetId)) {
      console.log(`Tweet ${tweetId} already processed, skipping...`);
      return;
    }

    try {
      console.log(
        `Processing tweet ${tweetId} from @${authorUsername}: ${tweetText}`
      );
      this.processedTweets.add(tweetId);

      const prompt = 
        `User @${authorUsername} tweeted: "${tweetText}"\n\n` +
        `Please provide a helpful response that could be sent as a reply. ` +
        `Keep it short, engaging, and use emojis where appropriate. ` +
        `Do not use markdown or HTML.`;

      const response = await this.callLLM(prompt);
      console.log("Agent response:", response);

      // Send the response as a reply
      if (response && response.trim()) {
        try {
          const result = await this.twitterClient.v2.reply(response, tweetId);
          console.log(`Reply sent: ${result.data.id}`);
        } catch (error) {
          console.error("Error sending reply:", error);
        }
      }
    } catch (error) {
      console.error("Error handling tweet:", error);
    }
  }

  private async generateAutonomousAction(): Promise<string> {
    // Weight the prompts - higher numbers mean more frequent selection
    const weightedPrompts = [
      // Informative content (weight: 4)
      {
        prompt:
          "Share an interesting fact or insight about Base blockchain or Layer 2 solutions",
        weight: 4,
      },
      {
        prompt: "Discuss a recent development or trend in the crypto ecosystem",
        weight: 4,
      },
      {
        prompt: "Explain a basic crypto concept in a simple, engaging way",
        weight: 4,
      },
      {
        prompt: "Share tips about web3 development or using CDP tools",
        weight: 4,
      },

      // Community engagement (weight: 3)
      {
        prompt: "Start a discussion about the future of DeFi or NFTs",
        weight: 3,
      },
      {
        prompt: "Ask the community about their favorite web3 tools or projects",
        weight: 3,
      },
      {
        prompt: "Share an interesting use case of blockchain technology",
        weight: 3,
      },
      { prompt: "Highlight a cool feature of Base or CDP", weight: 3 },

      // Project updates (weight: 2)
      { prompt: "Share what you can do as an AI agent on Base", weight: 2 },
      { prompt: "Explain one of your capabilities or tools", weight: 2 },
      { prompt: "Share a success story or interesting interaction", weight: 2 },

      // Rare on-chain actions (weight: 1)
      {
        prompt: "Deploy a creative meme token with an interesting concept",
        weight: 1,
      },
      {
        prompt: "Create an NFT collection about current crypto trends",
        weight: 1,
      },
    ];

    // Calculate total weight
    const totalWeight = weightedPrompts.reduce(
      (sum, item) => sum + item.weight,
      0
    );

    // Generate random number between 0 and total weight
    let random = Math.random() * totalWeight;

    // Find the selected prompt based on weights
    for (const { prompt, weight } of weightedPrompts) {
      random -= weight;
      if (random <= 0) {
        return prompt;
      }
    }

    // Fallback to first prompt (should never happen)
    return weightedPrompts[0].prompt;
  }

  async runAutonomousMode(interval = 3600) {
    while (true) {
      try {
        const thought = await this.generateAutonomousAction();

        const prompt = 
          `Create an engaging tweet based on this idea: ${thought}\n\n` +
          `Guidelines:\n` +
          `- Focus on providing value through information and engagement\n` +
          `- Keep tweets concise and friendly (under 280 characters)\n` +
          `- Use emojis appropriately\n` +
          `- Include hashtags like #Base #Web3 when relevant\n` +
          `- Don't use markdown or HTML\n` +
          `Just provide the tweet text directly.`;

        const response = await this.callLLM(prompt);
        console.log("Generated tweet:", response);

        // Send the tweet
        if (response && response.trim()) {
          try {
            const result = await this.twitterClient.v2.tweet(response);
            console.log(`Tweet sent: ${result.data.id}`);
          } catch (error) {
            console.error("Error sending tweet:", error);
          }
        }

        await new Promise((resolve) => setTimeout(resolve, interval * 1000));
      } catch (error) {
        console.error("Error in autonomous mode:", error);
        await new Promise((resolve) => setTimeout(resolve, 60 * 1000));
      }
    }
  }

  private async checkMentions() {
    try {
      // Get recent mentions using search API
      const mentions = await this.twitterClient.v2.userMentionTimeline(
        process.env.TWITTER_USER_ID!, // Your bot's user ID
        {
          "tweet.fields": ["author_id", "referenced_tweets"],
          expansions: ["author_id"],
          max_results: 10,
          ...(this.lastProcessedMentionId && {
            since_id: this.lastProcessedMentionId,
          }),
        }
      );

      // console.log("Mentions:", mentions);

      // Check if mentions exists and has tweets
      if (mentions && Array.isArray(mentions.tweets)) {
        for (const tweet of mentions.tweets) {
          // Skip if it's a retweet
          const isRetweet = tweet.referenced_tweets?.some(
            (ref: { type: string }) => ref.type === "retweet"
          );
          if (isRetweet) continue;

          const author = mentions.includes?.users?.find(
            (user: { id: string }) => user.id === tweet.author_id
          );

          if (author) {
            // Pass tweet ID to handleTweet
            await this.handleTweet(tweet.text, author.username, tweet.id);
            // Add small delay between processing mentions to avoid rate limits
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
        }
      }
    } catch (error) {
      console.error("Error checking mentions:", error);
    }
  }

  // Replace startListening with pollMentions
  async pollMentions(interval = 1200) {
    // Check every 20 minutes
    console.log("Started polling for mentions...");

    while (true) {
      await this.checkMentions();
      await new Promise((resolve) => setTimeout(resolve, interval * 1000));
    }
  }
}
