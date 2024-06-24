
## Peer to Peer Energy Trading Simulator (In Progress)

- This simulator follows the openAI gym-like environment style.
- The simulation dataset is Household energy demand and PV generation from AusGrid.
- Some households might remain PV generations in daytime, since it has large pv modules or not uses electrical energy because of absences.
- At this time, the households can play as energy seller and other household can buy from the energy.
- Market Operator manages the P2P market process, matching sellers and buyes in order to optimize their profits.
-------

### Simulator Design (Must be revised, some contents have not been written)

----
- AuctionEnv Contains AuctionController which controls Model and View and interfaces with AuctionEnv
- Market Operator including Participants get actions to participate auction, processing bidding.
- AuctionContext can make auction mechanism such as double-sided, uniform auction. 
- AuctionScenario determines the information that each customer can be got to make decision participating auction market.

```mermaid
classDiagram
    class AuctionEnv {
      +AuctionModel model
      +AuctionView view
      +AuctionController controller
      +step(action)
      +reset()
      +render(mode)
    }

    class DataContainer {
      +DataPath data_path
      +DataSet dataset
      +_add_month()
      +get_size()
      +get_customer_ids()
    }

    class AuctionModel {
      +DataContainer data_container
      +AuctionScenario scenario
      +reset()
      +run_auction()
      +get_state()
    }

    class AuctionScenario {
      +get_observation_space()
      +get_actiob_space()
      +get_state()
    }

    class AuctionContext {
      +Dict matched_order
      +Dict non_matched_order
      +make_bid_price()
      +get_orderbook()
      +transaction()
    }

    class AuctionController {
      +AuctionModel model
      +AuctionView view
      +Reward reward
      +get_size()
      +get_reward()
      +get_state()
      +reset()
      +get_observation_space()
      +get_action_space()
      +run_auction()
      +run_render()
    }

    class AuctionView {
      +render(state)
    }

    class MarketOperator {
      +List[Participant] participant
      +float trading_fee
      +get_actions()
      +_bid_process()
      +_get_orderbook(AuctionContext)
    }

    class Participant {
      +int id
      +bool has_product
      +bid_action()
    }

    class Reward {
      +get_reward()
    }

    AuctionEnv --> AuctionController : interfaces
    AuctionEnv --> AuctionContext : contains

    AuctionController --> AuctionModel : controls
    AuctionController --> AuctionView : controls
    AuctionController --> Reward : controls

    AuctionModel --> DataContainer : contains
    AuctionModel --> MarketOperator : uses
    AuctionModel --> AuctionScenario : uses

    MarketOperator --> Participant : contains
  
```
