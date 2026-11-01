import Foundation
import FoundationModels

@main
struct AppleIntelligenceApp {
    static func main() async {
        print("==================================================")
        print(" Apple Intelligence Client - Hybrid Session      ")
        print("==================================================")
        
        let session: LanguageModelSession
        let cloudModel = PrivateCloudComputeLanguageModel()
        
        // Attempt cloud routing if available and permitted, otherwise safely use on-device model
        if cloudModel.isAvailable {
            print("[*] Private Cloud Compute is reachable.")
        }
        
        // Use default system language model (safe, stable, runs on Apple Silicon unified memory)
        print("[*] Initializing LanguageModelSession with SystemLanguageModel.default...")
        session = LanguageModelSession()
        
        let prompt = "Explain step-by-step how Apple Silicon unified memory accelerates on-device AI inference."
        print("\n[>] Prompt: \(prompt)\n")
        print("[*] Generating response...")
        
        do {
            let response = try await session.respond(to: prompt)
            print("\n--- [Apple Foundation Model Response] ---")
            print(response.content)
            print("------------------------------------------")
            print("\nTelemetry:")
            print("  Total Output Tokens: \(response.usage.output.totalTokenCount)")
            print("  Reasoning Tokens:    \(response.usage.output.reasoningTokenCount)")
        } catch {
            print("[-] Model response error: \(error)")
        }
    }
}
